"""Command line entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from .ablator import ObliqueAblator
from .data import (grounding_pairs, load_lines, match_sizes,
                    split_fit_eval)
from .directions import (build_directions, find_decoder_layers, load_model,
                         residual_means, separability)
from .export import (apply_to_weights, copy_tokenizer_files,
                     restore_missing_tensors, write_manifest)
from .metrics import count_refusals, first_token_logprobs
from .search import run_search


def build_parser():
    p = argparse.ArgumentParser(
        prog="oblique-abliterate",
        description="Abliteration with an oblique projector that preserves a "
                    "chosen behaviour direction.")
    p.add_argument("--model", required=True,
                   help="HuggingFace repo id (downloaded and cached) or a path "
                        "to a local model directory")
    p.add_argument("--out", default="", help="output dir (default: out/<name>-oblique)")
    p.add_argument("--study", default="", help="where to write the search record")

    g = p.add_argument_group("prompt sets")
    g.add_argument("--harmful", default="data/harmful.txt")
    g.add_argument("--harmless", default="data/harmless.txt")
    g.add_argument("--answerable", default="", help="default: generated in-process")
    g.add_argument("--unanswerable", default="")

    g = p.add_argument_group("method")
    g.add_argument("--preserve-grounding", action="store_true",
                   help="THE POINT OF THIS PACKAGE: use an oblique projector that "
                        "keeps the hedging direction. Without it you get ordinary "
                        "abliteration, which is useful only as a baseline.")
    g.add_argument("--preserve-rank", type=int, default=1,
                   help="dimensionality of the preserved subspace (1 is what was "
                        "validated; >1 is an untested experiment)")
    g.add_argument("--no-project", action="store_true",
                   help="skip orthogonalising the refusal direction against the "
                        "harmless direction")

    g = p.add_argument_group("search")
    g.add_argument("--n", type=int, default=256, help="prompts per side for fitting")
    g.add_argument("--n-eval", type=int, default=48, help="prompts for scoring")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--trials", type=int, default=40)
    g.add_argument("--startup-trials", type=int, default=12)
    g.add_argument("--kl-target", type=float, default=0.01)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="cpu",
                   help="cpu is the default on purpose: the whole model is held "
                        "in bf16 and most GPUs cannot fit a large one")
    g.add_argument("--dry-run", action="store_true",
                   help="report geometry and separability, then stop. Costs one "
                        "model load and no search; run this first.")
    g.add_argument("--export-from-study", action="store_true",
                   help="skip the search and export a previously recorded config")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = lambda *a: print("[oblique]", *a, flush=True)

    name = args.model.rstrip("/").split("/")[-1]
    tag = f"{name}-oblique" if args.preserve_grounding else f"{name}-abliterated"
    args.out = args.out or f"out/{tag}"
    args.study = args.study or f"out/{tag}.study.json"

    # Imported here, not at module scope, purely so `--help` and argument
    # errors return instantly. transformers takes several seconds to import.
    from transformers import AutoTokenizer

    harmful = load_lines(args.harmful)
    harmless = load_lines(args.harmless)
    # Hold the eval slice out FIRST, then fit on what is left. Slicing both
    # from the same list independently overlaps once a set is smaller than
    # n + n_eval, which scores the edit on prompts it was fitted on.
    pool_bad, ev_bad = split_fit_eval(harmful, args.n, args.n_eval)
    pool_good, ev_good = split_fit_eval(harmless, args.n, args.n_eval)
    fit_bad, fit_good, n = match_sizes(pool_bad, pool_good, args.n)
    log(f"{tag} | fit {n}+{n} | eval {len(ev_bad)}+{len(ev_good)}")

    ans = unans = None
    if args.preserve_grounding:
        if args.answerable and args.unanswerable:
            ans, unans = load_lines(args.answerable), load_lines(args.unanswerable)
            ans, unans, _ = match_sizes(ans, unans, n)
        else:
            ans, unans = grounding_pairs(n, seed=args.seed)
        log(f"grounding contrast: {len(unans)} unanswerable / {len(ans)} answerable")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # generation needs left padding

    log("loading model in bf16 (the slow part)...")
    model = load_model(args.model, torch.bfloat16, args.device, log=log)
    layers = find_decoder_layers(model)
    log(f"{len(layers)} decoder layers")

    log("residual means (harmless)...")
    good_means = residual_means(model, tok, fit_good, args.batch, args.device, log=log)
    log("residual means (harmful)...")
    bad_means = residual_means(model, tok, fit_bad, args.batch, args.device, log=log)
    dirs = build_directions(good_means, bad_means, project=not args.no_project)

    preserve_dirs = None
    if args.preserve_grounding:
        log("residual means (answerable)...")
        ans_means = residual_means(model, tok, ans, args.batch, args.device, log=log)
        log("residual means (unanswerable)...")
        unans_means = residual_means(model, tok, unans, args.batch, args.device, log=log)
        # Points toward "I should hedge".
        preserve_dirs = build_directions(ans_means, unans_means, project=False)
        sep = separability(dirs, preserve_dirs)
        log(f"|cos(refusal, hedging)| median over upper half = "
            f"{sep['median_abs_cos_upper_half']:.3f}  "
            f"(low means separable and the projector is well conditioned)")
        if sep["median_abs_cos_upper_half"] > 0.5:
            log("WARNING: the two behaviours are strongly aligned on this model. "
                "The oblique edit will be heavily amplified and may not help.")

    if args.dry_run:
        log("dry run: geometry only, stopping before search")
        return 0

    if args.export_from_study:
        best = json.load(open(args.study))["best"]
        return _export(args, model, layers, dirs, preserve_dirs, best, log)

    log("baseline (unmodified)...")
    base_lp = first_token_logprobs(model, tok, ev_good, args.batch, args.device)
    base_ref = count_refusals(model, tok, ev_bad, args.batch, args.device)
    log(f"baseline refusals {base_ref}/{len(ev_bad)}")
    if base_ref == 0:
        log("WARNING: this model already refuses nothing on your harmful set, so "
            "there is no signal to optimise against.")

    ablator = ObliqueAblator(model, layers).install()
    try:
        best, results = run_search(
            model, tok, layers, ablator, dirs, preserve_dirs,
            eval_harmful=ev_bad, eval_harmless=ev_good,
            baseline_logprobs=base_lp, baseline_refusals=base_ref,
            trials=args.trials, startup_trials=args.startup_trials,
            batch=args.batch, device=args.device, kl_target=args.kl_target,
            preserve_rank=args.preserve_rank, seed=args.seed, log=log)
    finally:
        ablator.remove()

    log(f"BEST: refusals {best['refusals']}/{len(ev_bad)} "
        f"(baseline {base_ref}) at KL {best['kl']:.4f}")

    sp = Path(args.study)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"model": args.model, "baseline_refusals": base_ref,
                              "preserve_grounding": args.preserve_grounding,
                              "trials": results, "best": best}, indent=2))
    log(f"study -> {sp}")
    return _export(args, model, layers, dirs, preserve_dirs, best, log)


def _export(args, model, layers, dirs, preserve_dirs, best, log):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"exporting -> {out_dir}")

    edited, skipped, lev = apply_to_weights(layers, dirs, best,
                                            preserve_dirs=preserve_dirs, log=log)
    if edited == 0:
        log("ERROR: no matrices were edited. The writer selection did not match "
            "this architecture; the output would be an unmodified copy.")
        return 1

    model.save_pretrained(str(out_dir), safe_serialization=True)

    src = Path(args.model)
    if not src.exists():
        src = Path(snapshot_download(args.model,
                                     allow_patterns=["*.safetensors", "*.json",
                                                     "tokenizer*", "vocab*",
                                                     "merges*", "*.jinja",
                                                     "*preprocessor*"]))
    try:
        restore_missing_tensors(src, out_dir, log=log)
    except Exception as e:
        log(f"WARNING: could not restore dropped tensors: {e}")
    try:
        copy_tokenizer_files(src, out_dir, log=log)
    except Exception as e:
        log(f"WARNING: could not copy tokenizer files: {e}")

    write_manifest(out_dir, args.model, best, preserve_dirs is not None, edited, lev)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
