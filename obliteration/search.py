"""Two-objective TPE search over the edit parameters.

Nine parameters: one continuous direction index, plus four shaping the
triangular depth kernel (max_w, pos, min_frac, dist) for each of attention and
MLP. Scored jointly on refusals removed and KL from the unmodified model.

The search bounds below carry the assumption, from Arditi et al., "Refusal in
Language Models Is Mediated by a Single Direction" (arXiv:2406.11717), that
refusal is mediated slightly past the midpoint of the stack. Sampling the full depth range let the kernel peak at layer 0 and
produced KL between 1.0 and 2.5, which is a destroyed model, where the bounded
search reaches KL well under 0.1.
"""
from __future__ import annotations

import optuna
from optuna.samplers import TPESampler

from .ablator import COMPONENTS
from .metrics import count_refusals, first_token_logprobs, kl_from_baseline


def suggest_params(trial, n_layers: int) -> tuple:
    """Sample one candidate configuration."""
    last = n_layers - 1.0
    direction_index = trial.suggest_float("direction_index", 0.4 * last, 0.9 * last)
    params = {}
    for comp in COMPONENTS:
        max_w = trial.suggest_float(f"{comp}.max_w", 0.8, 1.5)
        # min_w is sampled as a FRACTION of max_w so it can never exceed it.
        # Sampling the two independently inverted the triangle, making the edit
        # stronger further from the peak, which is the opposite of the intent.
        min_frac = trial.suggest_float(f"{comp}.min_frac", 0.0, 1.0)
        params[comp] = {
            "max_w": max_w,
            "pos": trial.suggest_float(f"{comp}.pos", 0.6 * last, 1.0 * last),
            "min_w": min_frac * max_w,
            "dist": trial.suggest_float(f"{comp}.dist", 1.0, 0.6 * last),
        }
    return direction_index, params


def run_search(model, tok, layers, ablator, dirs, preserve_dirs, *,
               eval_harmful, eval_harmless, baseline_logprobs, baseline_refusals,
               trials: int, startup_trials: int, batch: int, device: str,
               kl_target: float, preserve_rank: int, seed: int, log=print):
    """Optimise and return (best, all_results)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []

    def objective(trial):
        direction_index, params = suggest_params(trial, len(layers))
        ablator.configure(dirs, direction_index, params,
                          preserve_dirs=preserve_dirs, preserve_rank=preserve_rank)

        lp = first_token_logprobs(model, tok, eval_harmless, batch, device)
        kl = kl_from_baseline(lp, baseline_logprobs)
        refusals = count_refusals(model, tok, eval_harmful, batch, device)
        ablator.reset()

        refusal_score = (refusals / baseline_refusals) if baseline_refusals else float(refusals)
        # Once KL is under target, stop rewarding further reduction and let the
        # optimiser spend its remaining budget on removing refusals.
        kl_score = kl if kl >= kl_target else refusal_score * kl_target

        log(f"trial {trial.number}: refusals {refusals}/{len(eval_harmful)} KL {kl:.4f}")
        results.append({"trial": trial.number, "refusals": refusals, "kl": kl,
                        "direction_index": direction_index, "params": params})
        return kl_score, refusal_score

    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=TPESampler(seed=seed, n_startup_trials=startup_trials,
                           multivariate=True, group=True))
    study.optimize(objective, n_trials=trials)

    # Prefer configurations that did not wreck the model. If every trial did,
    # fall back to the whole pool so the caller still gets a result plus a
    # visible warning rather than an exception.
    if not results:
        raise SystemExit("the search recorded no trials; nothing to export")
    usable = [r for r in results if r["kl"] < 0.5]
    if not usable:
        log("WARNING: every trial exceeded KL 0.5. The chosen config is the least "
            "bad of a bad set; treat the output as damaged and re-run with a "
            "lower --kl-target or a narrower depth band.")
    best = min(usable or results, key=lambda r: (r["refusals"], r["kl"]))
    best["preserve_rank"] = preserve_rank
    return best, results
