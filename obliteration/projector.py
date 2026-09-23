"""The oblique projector: the one piece that distinguishes this from ordinary
abliteration.

Standard abliteration removes a refusal direction r with a symmetric projection,
y' = y - w (r.y) r. That uses r for two different jobs at once: it is both the
thing being removed AND the ruler used to measure how much of it is present. The
side effect is that it protects exactly r's orthogonal complement, which nobody
chose and which is usually the wrong thing to protect.

An oblique projector splits the two jobs:

    y' = y - w (u.y) r        with   u.r = 1,  u.j = 0

r is still what gets removed. u is a separate covector chosen so that a direction
j you want to keep reads as exactly zero. Refusal is removed in full; j passes
through untouched.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def oblique_covector(r: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Build u such that u.r = 1 and u.j = 0 for every j in span(J).

    J is [k, hidden], or a single [hidden] vector. Passing more than one row
    preserves a whole subspace: the constraint is solved directly by least
    squares, so u.j is exactly zero for every j in the span rather than small.

        u0 = r - J^+ (J r)      component of r orthogonal to span(J)
        u  = u0 / (u0 . r)      rescale so it reads exactly one unit of r

    k=1 reduces to the plain difference-of-means preservation.

    Raises ValueError if r lies inside span(J). That is not a numerical edge
    case to paper over: it is the measurable statement that, at this layer, the
    two behaviours are not linearly separable, so no projector can remove one
    while preserving the other. Callers are expected to fall back to u = r and
    accept the collateral damage at that layer rather than pretend otherwise.
    """
    # Checked before the solve: a non-finite input reaches linalg.lstsq and
    # surfaces as an INTERNAL ASSERT from the backend, which reads like a torch
    # bug rather than bad activations. It also slips past the denominator guard
    # below, since NaN < 1e-6 is False, so a silent all-NaN covector would be
    # returned and written into the weights.
    if not torch.isfinite(r).all():
        raise ValueError("refusal direction contains non-finite values")
    if not torch.isfinite(J).all():
        raise ValueError("preserved direction contains non-finite values")

    r = F.normalize(r, p=2, dim=0)
    B = J if J.dim() == 2 else J.unsqueeze(0)          # [k, hidden]
    B = F.normalize(B, p=2, dim=1)

    # Least squares gives the component of r inside span(B) for a possibly
    # non-orthonormal B. Solved on CPU in fp32 on purpose: ROCm's linalg.lstsq
    # backend raises (INTERNAL ASSERT / illegal argument 4), and this system is
    # [hidden x k] with k small, so the copy costs nothing.
    dev, dt = r.device, r.dtype
    Bc, rc = B.detach().cpu().float(), r.detach().cpu().float()
    coef = torch.linalg.lstsq(Bc.T, rc.unsqueeze(-1)).solution.squeeze(-1)
    u0 = (rc - Bc.T @ coef).to(device=dev, dtype=dt)

    denom = torch.dot(u0, r)
    if denom.abs() < 1e-6:
        raise ValueError(
            f"refusal lies inside the preserved subspace (u0.r={denom:.2e}); "
            "no oblique projector exists at this layer")
    return u0 / denom


def preserve_subspace(preserve_dirs: torch.Tensor, direction_index: float,
                      k: int = 1) -> torch.Tensor:
    """Build the [k, hidden] basis of directions to protect, at a continuous
    layer index.

    k=1 is the interpolated preserve direction and is what every result in the
    README used. k>1 needs several independent directions; the search path only has one
    mean direction per layer, so it uses this layer's plus its neighbours' as a
    cheap proxy basis, orthonormalised by QR. That proxy is untested and is
    offered as an experiment, not a recommendation.
    """
    j = interpolate_direction(preserve_dirs, direction_index)
    if k <= 1:
        return j.unsqueeze(0)
    frac, idx = math.modf(direction_index + 1)
    base = int(idx)
    rows = [base + o for o in range(-(k // 2), k - (k // 2))
            if 0 <= base + o < preserve_dirs.shape[0]]
    B = F.normalize(preserve_dirs[rows], p=2, dim=1)
    return torch.linalg.qr(B.T)[0].T                   # orthonormal [<=k, hidden]


def interpolate_direction(dirs: torch.Tensor, direction_index: float) -> torch.Tensor:
    """Continuous layer index: 15.81 means 19% of the way from layer 15 to 16.

    Row 0 of `dirs` is the embedding output, hence the +1.
    """
    frac, idx = math.modf(direction_index + 1)
    i = int(idx)
    j = min(i + 1, dirs.shape[0] - 1)
    return F.normalize(dirs[i].lerp(dirs[j], frac), p=2, dim=0)


def leverage(r: torch.Tensor, u: torch.Tensor) -> float:
    """||u|| / ||r||: how much the oblique edit is amplified versus the blunt one.

    This is the cost of the method and it is bounded by 1 / sin(angle(r, j)).
    At the measured cos(r, j) = 0.18 on Qwen3-4B layer 21 it is about 1.017, so
    effectively free. It grows without limit as the two behaviours approach
    parallel, which is the same condition oblique_covector refuses outright.
    Report it; a large value means the edit is straining.
    """
    return (u.norm() / r.norm()).item()
