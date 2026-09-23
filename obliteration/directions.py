"""Extracting behaviour directions from a model's residual stream.

A "direction" here is a unit vector in hidden space obtained by contrasting two
matched sets of prompts. Refusal is (harmful - harmless). Hedging is
(unanswerable - answerable). The same machinery produces both; only the prompt
sets differ.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def residual_means(model, tok, prompts, batch: int, device: str,
                   max_length: int = 128, log=print) -> torch.Tensor:
    """Mean last-token hidden state at every layer, as [n_layers + 1, hidden].

    The last token of the prompt is used because that is the position the model
    is about to generate from, so it carries the decision about how to answer.
    """
    total, count = None, 0
    for i in range(0, len(prompts), batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        idx = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(idx.size(0), device=idx.device)
        stack = torch.stack([h[rows, idx].float() for h in out.hidden_states], dim=1)

        # Drop non-finite rows rather than let one bad prompt poison a whole
        # layer's mean with NaN. Seen in practice with the ROCm attention kernel
        # at scale; cheap insurance everywhere else.
        finite = torch.isfinite(stack).all(dim=2).all(dim=1)
        stack = stack[finite]
        if stack.size(0) == 0:
            del out
            log(f"  {min(i + batch, len(prompts))}/{len(prompts)} (all non-finite, skipped)")
            continue

        s = stack.sum(dim=0).cpu()
        total = s if total is None else total + s
        count += stack.size(0)
        del out, stack
        log(f"  {min(i + batch, len(prompts))}/{len(prompts)}")

    if total is None or count == 0:
        raise RuntimeError("every prompt produced non-finite activations")
    return total / count


def build_directions(neutral_means: torch.Tensor, behaviour_means: torch.Tensor,
                     project: bool = True) -> torch.Tensor:
    """One unit direction per layer: behaviour minus neutral.

    project=True additionally removes the component lying along the neutral
    representation itself (the orthogonalisation attributed to grimjim), so the edit
    subtracts only what is specific to the behaviour rather than dragging
    general capability along with it.

        r <- normalize(r - (r . g) g)

    Matched set sizes matter more than they look. Unmatched sets make the
    difference of means capture dataset identity instead of behaviour; with
    AdvBench against Alpaca that was measured as roughly a 35 degree rotation
    away from the true refusal direction.
    """
    dirs = F.normalize(behaviour_means - neutral_means, p=2, dim=1)
    if project:
        neutral = F.normalize(neutral_means, p=2, dim=1)
        comp = torch.sum(dirs * neutral, dim=1, keepdim=True)
        dirs = F.normalize(dirs - comp * neutral, p=2, dim=1)
    return dirs


def separability(refusal_dirs: torch.Tensor, preserve_dirs: torch.Tensor) -> dict:
    """How separable the two behaviours are, per layer and over the upper half.

    The oblique projector is well conditioned exactly when these directions are
    far from parallel. Near 0 means two genuinely distinct behaviours and a
    nearly free edit. Near 1 means they are entangled, and no projector can
    remove one while keeping the other.

    Reported before the search so the method can be ruled out cheaply.
    """
    cos = F.cosine_similarity(refusal_dirs, preserve_dirs, dim=1)
    upper = cos[len(cos) // 2:]
    return {
        "per_layer_abs_cos": cos.abs().tolist(),
        "median_abs_cos_upper_half": upper.abs().median().item(),
        "max_abs_cos_upper_half": upper.abs().max().item(),
    }


def is_multimodal_config(config) -> bool:
    """True when the text model is nested under `config.text_config`.

    Vision-capable checkpoints keep vocab_size, hidden_size and the rest inside
    a sub-config, and declare a ...ForConditionalGeneration architecture. Asking
    AutoModelForCausalLM for one of those fails with a bare
    `'...Config' object has no attribute 'vocab_size'`, which says nothing about
    the real cause.
    """
    if getattr(config, "text_config", None) is not None:
        return True
    return any(str(a).endswith("ForConditionalGeneration")
               for a in (getattr(config, "architectures", None) or []))


def load_model(model_id, dtype, device: str, log=print):
    """Load a checkpoint, picking the auto class its config actually maps to.

    `model_id` is either a HuggingFace repo id, which is downloaded and cached,
    or a path to a local directory.

    Only the decoder stack is edited either way. Loading the multimodal class
    keeps the vision tower attached, which is harmless and means it survives
    the export intact.
    """
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoModelForImageTextToText)
    config = AutoConfig.from_pretrained(model_id)
    if is_multimodal_config(config):
        log("multimodal config detected; loading the vision-capable class")
        cls = AutoModelForImageTextToText
    else:
        cls = AutoModelForCausalLM
    return cls.from_pretrained(model_id, dtype=dtype,
                               device_map={"": device},
                               low_cpu_mem_usage=True).eval()


def find_decoder_layers(model) -> torch.nn.ModuleList:
    """Locate the decoder layer stack across the common wrapper layouts."""
    m = model
    for attr in ("model", "language_model", "layers"):
        if hasattr(m, attr):
            m = getattr(m, attr)
        if isinstance(m, torch.nn.ModuleList):
            return m
    raise RuntimeError(
        "could not locate decoder layers; pass the stack explicitly. This "
        "package has only been run on Qwen layouts.")
