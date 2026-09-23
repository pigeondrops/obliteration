"""Applying a candidate edit during search, via forward hooks.

The search needs to try hundreds of configurations. Writing weights each time
would be unusably slow, so candidates are applied as forward hooks on the blocks
that write into the residual stream. This is not an approximation. For y = Wx:

    (W - w r u^T W) x  ==  y - w (u.y) r

so hooking the output is exactly the weight edit, costs no extra memory, and
resets instantly. The real weight edit happens once, at export.

Hooking the BLOCK output rather than individual projection modules is what makes
this work on a fused-expert MoE. In that layout `mlp.experts.down_proj` is a
single 3-D Parameter holding every expert, not one Module per expert, so there
is no submodule to hook. But projection is linear, so ablating the summed block
output equals ablating every writing matrix inside it:

    P(sum_e down_proj_e(x) + shared(x)) == sum_e P(down_proj_e(x)) + P(shared(x))

One hook per block therefore covers all experts at once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .projector import interpolate_direction, oblique_covector, preserve_subspace

# The two sublayers that write into the residual stream. Each gets its own
# fitted depth curve, because attention and MLP do not carry refusal equally.
COMPONENTS = {
    "attn": ("self_attn", "linear_attn"),
    "mlp": ("mlp",),
}

_ALL_SUFFIXES = tuple(s for sfxs in COMPONENTS.values() for s in sfxs)


def layer_weight(layer_index: int, p: dict):
    """Triangular kernel over depth: full strength at p['pos'], decaying out.

        weight(L) = max_w + (|L - pos| / dist) * (min_w - max_w)

    Returns None beyond `dist`, meaning that layer is left completely alone.
    """
    d = abs(layer_index - p["pos"])
    if d > p["dist"]:
        return None
    return p["max_w"] + (d / max(p["dist"], 1e-6)) * (p["min_w"] - p["max_w"])


class ObliqueAblator:
    """Applies y <- y - w (u.y) r on every residual-writing block.

    With preserve_dirs=None this degrades exactly to standard abliteration
    (u = r), which makes it a drop-in baseline for A/B comparison.
    """

    def __init__(self, model, layers):
        self.model, self.layers = model, layers
        self.handles = []
        self._cfg = {}                  # id(module) -> (r, u, w)
        self.last_leverage = []         # per-layer ||u||/||r||, for reporting
        self.inseparable_layers = []    # layers where no projector existed

    def _hook(self, module, _inp, output):
        cfg = self._cfg.get(id(module))
        if cfg is None:
            return output
        r, u, w = cfg

        def ab(t):
            rr = r.to(t.dtype).to(t.device)
            uu = u.to(t.dtype).to(t.device)
            return t - w * (t @ uu).unsqueeze(-1) * rr     # remove r, measure u

        if isinstance(output, tuple):
            return (ab(output[0]),) + output[1:]
        return ab(output)

    def install(self):
        for layer in self.layers:
            # Direct children only: we want the block, not modules inside it.
            for name, mod in layer.named_children():
                if name in _ALL_SUFFIXES:
                    self.handles.append(mod.register_forward_hook(self._hook))
        return self

    def configure(self, dirs, direction_index, params,
                  preserve_dirs=None, preserve_rank: int = 1):
        """Set the (r, u, w) triple for every hooked module. Instant."""
        self._cfg.clear()
        self.last_leverage.clear()
        self.inseparable_layers.clear()

        fixed = (interpolate_direction(dirs, direction_index)
                 if direction_index is not None else None)

        for li, layer in enumerate(self.layers):
            r = fixed if fixed is not None else F.normalize(dirs[li + 1], p=2, dim=0)
            if preserve_dirs is None:
                u = r                                   # standard abliteration
            else:
                idx = direction_index if direction_index is not None else float(li)
                J = preserve_subspace(preserve_dirs, idx, preserve_rank)
                try:
                    u = oblique_covector(r, J)
                    self.last_leverage.append((u.norm() / r.norm()).item())
                except ValueError:
                    # Inseparable at this layer. Fall back to the blunt edit and
                    # record it, rather than silently skipping the layer.
                    u = r
                    self.inseparable_layers.append(li)

            for comp, names in COMPONENTS.items():
                w = layer_weight(li, params[comp])
                if w is None:
                    continue
                for name, mod in layer.named_children():
                    if name in names:
                        self._cfg[id(mod)] = (r, u, w)

    def reset(self):
        self._cfg.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False
