"""Writing the edit into real weights, once, after the search has chosen a config.

The search hooks block outputs, which covers everything inside them. The export
has to touch the actual writing matrices, including `mlp.experts.down_proj`,
which on a fused MoE is a single 3-D Parameter holding every expert rather than
one Module per expert. On the model this was built for, that tensor holds most
of the parameters that write the refusal direction, so the export indexes into
it expert by expert.
"""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .ablator import COMPONENTS, layer_weight
from .projector import interpolate_direction, oblique_covector, preserve_subspace

ATTN_LEAVES = ("o_proj", "out_proj")
MLP_LEAVES = ("down_proj",)


def _project_(t: torch.Tensor, r: torch.Tensor, u: torch.Tensor, w: float):
    """W <- W - w * outer(r, u @ W), on whichever axis matches r.

    Returns the edited tensor, or None if no axis matches (caller counts it as
    skipped rather than guessing). fp32 throughout, cast back at the end, so a
    bf16 checkpoint does not accumulate rounding across layers.
    """
    if t.dim() == 2 and t.shape[0] == r.shape[0]:
        tf = t.to(torch.float32)
        tf -= w * torch.outer(r, u @ tf)
        return tf.to(t.dtype)

    if t.dim() == 3:
        # Fused expert stacks, in both index orders.
        if t.shape[1] == r.shape[0]:                      # [E, hidden, d_ff]
            tf = t.to(torch.float32)
            for e in range(tf.shape[0]):
                tf[e] -= w * torch.outer(r, u @ tf[e])
            return tf.to(t.dtype)
        if t.shape[2] == r.shape[0]:                      # [E, d_ff, hidden]
            tf = t.to(torch.float32)
            for e in range(tf.shape[0]):
                m = tf[e].T.contiguous()
                m -= w * torch.outer(r, u @ m)
                tf[e] = m.T
            return tf.to(t.dtype)
    return None


@torch.no_grad()
def apply_to_weights(layers, dirs, best, preserve_dirs=None, log=print):
    """Apply the winning config to the real parameters in place.

    Returns (edited, skipped, leverage). `skipped` counts matrices whose shape
    matched no axis; a nonzero value on a new architecture means the writer
    selection needs checking, not that the edit silently half-happened.
    """
    r_fixed = interpolate_direction(dirs, best["direction_index"])
    if preserve_dirs is not None:
        J = preserve_subspace(preserve_dirs, best["direction_index"],
                              best.get("preserve_rank", 1))
        try:
            u_fixed = oblique_covector(r_fixed, J)
        except ValueError:
            log("WARNING: refusal and preserved direction are inseparable at the "
                "chosen layer; falling back to a standard (non-oblique) edit")
            u_fixed = r_fixed
    else:
        u_fixed = r_fixed

    lev = (u_fixed.norm() / r_fixed.norm()).item()
    edited = skipped = 0

    for li, layer in enumerate(layers):
        for comp, leaves in (("attn", ATTN_LEAVES), ("mlp", MLP_LEAVES)):
            w = layer_weight(li, best["params"][comp])
            if w is None:
                continue
            for name, param in layer.named_parameters():
                stem = name[:-len(".weight")] if name.endswith(".weight") else name
                if stem.split(".")[-1] not in leaves:
                    continue
                out = _project_(param.data, r_fixed.to(param.device),
                                u_fixed.to(param.device), w)
                if out is None:
                    skipped += 1
                else:
                    param.data = out
                    edited += 1

    log(f"edited {edited} matrices ({skipped} shape-mismatched), leverage {lev:.4f}x")
    return edited, skipped, lev


@torch.no_grad()
def restore_missing_tensors(src_dir, out_dir, max_shard_gb: float = 5.0, log=print):
    """Copy back tensors that save_pretrained dropped.

    AutoModelForCausalLM loads only the text model, so saving omits the vision
    tower and any multi-token-prediction head. That makes the export unusable in
    two concrete ways: a GGUF vision conversion has no tensors to convert, and
    llama.cpp rejects the text model outright because the config still declares
    an MTP layer.

    Neither is touched by abliteration, since neither is in the decoder stack, so
    copying them verbatim yields a complete model.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    src_index = src_dir / "model.safetensors.index.json"
    out_index = out_dir / "model.safetensors.index.json"
    if not src_index.exists() or not out_index.exists():
        log("no shard index; skipping tensor restore")
        return 0

    src_map = json.load(open(src_index))["weight_map"]
    meta = json.load(open(out_index))
    out_map = meta["weight_map"]
    missing = [k for k in src_map if k not in out_map]
    if not missing:
        log("export is already complete")
        return 0

    kinds = {}
    for k in missing:
        kind = "vision" if "visual" in k else ("mtp" if "nextn" in k or "mtp" in k else "other")
        kinds[kind] = kinds.get(kind, 0) + 1
    log(f"restoring {len(missing)} dropped tensors {kinds}")

    by_shard = {}
    for k in missing:
        by_shard.setdefault(src_map[k], []).append(k)

    buf, buf_bytes, shard_no, total = {}, 0, 0, 0
    cap = max_shard_gb * 1e9

    def flush():
        nonlocal buf, buf_bytes, shard_no, total
        if not buf:
            return
        shard_no += 1
        name = f"model-restored-{shard_no:05d}.safetensors"
        save_file(buf, str(out_dir / name), metadata={"format": "pt"})
        for k, t in buf.items():
            out_map[k] = name
            total += t.numel() * t.element_size()
        buf, buf_bytes = {}, 0

    for shard, keys in by_shard.items():
        tensors = load_file(str(src_dir / shard))
        for k in keys:
            t = tensors[k]
            buf[k] = t
            buf_bytes += t.numel() * t.element_size()
            if buf_bytes >= cap:
                flush()
        del tensors
        gc.collect()
    flush()

    meta.setdefault("metadata", {})
    meta["metadata"]["total_size"] = meta["metadata"].get("total_size", 0) + total
    out_index.write_text(json.dumps(meta, indent=2))

    # Once the vision tower is back, the config must name the multimodal
    # architecture again or downstream converters refuse the model.
    try:
        src_cfg = json.load(open(src_dir / "config.json"))
        out_cfg_path = out_dir / "config.json"
        out_cfg = json.load(open(out_cfg_path))
        if src_cfg.get("architectures") and \
                src_cfg["architectures"] != out_cfg.get("architectures"):
            out_cfg["architectures"] = src_cfg["architectures"]
            for k in ("model_type", "vision_config", "image_token_id",
                      "video_token_id", "text_config"):
                if k in src_cfg and k not in out_cfg:
                    out_cfg[k] = src_cfg[k]
            out_cfg_path.write_text(json.dumps(out_cfg, indent=2))
            log(f"config architectures -> {src_cfg['architectures']}")
    except Exception as e:
        log(f"WARNING: could not restore config architectures: {e}")

    log(f"restored {len(missing)} tensors ({total / 1e9:.1f} GB) into {shard_no} shard(s)")
    return len(missing)


def copy_tokenizer_files(src_dir, out_dir, log=print):
    """Copy tokenizer/template files so the export is directly convertible.

    Without these, GGUF conversion falls back to a slow tokenizer and fails.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    n = 0
    for f in src_dir.iterdir():
        if f.name.startswith(("tokenizer", "vocab", "merges", "preprocessor")) \
                or f.suffix == ".jinja":
            shutil.copy2(f, out_dir / f.name)
            n += 1
    log(f"copied {n} tokenizer/template files")
    return n


def write_manifest(out_dir, model, best, preserve, edited, leverage, extra=None):
    """Record exactly what was done, next to the weights.

    Without this, an edited model carries no record of the direction, depth
    curve or strengths that produced it, and the run cannot be reproduced.
    """
    payload = {
        "method": "oblique abliteration",
        "model": model,
        "preserve_grounding": bool(preserve),
        "direction_index": best["direction_index"],
        "params": best["params"],
        "preserve_rank": best.get("preserve_rank", 1),
        "refusals_after": best.get("refusals"),
        "kl": best.get("kl"),
        "matrices_edited": edited,
        "oblique_leverage": leverage,
    }
    if extra:
        payload.update(extra)
    (Path(out_dir) / "abliteration.json").write_text(json.dumps(payload, indent=2))
    return payload
