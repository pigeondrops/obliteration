"""The oblique projector's defining properties, checked numerically.

The whole method rests on three claims: u reads exactly one unit of refusal, it
reads exactly zero of the preserved direction, and the resulting edit removes r
while leaving j untouched. Each is asserted here rather than argued in prose.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from obliteration.projector import (interpolate_direction, leverage,
                                     oblique_covector, preserve_subspace)


def _pair(cos_target, hidden=64, seed=0):
    """Two unit vectors with a prescribed cosine between them."""
    g = torch.Generator().manual_seed(seed)
    r = F.normalize(torch.randn(hidden, generator=g), dim=0)
    q = torch.randn(hidden, generator=g)
    q = F.normalize(q - (q @ r) * r, dim=0)          # orthogonal to r
    j = F.normalize(cos_target * r + math.sqrt(1 - cos_target ** 2) * q, dim=0)
    return r, j


@pytest.mark.parametrize("cos", [0.0, 0.05, 0.18, 0.4, 0.7])
def test_covector_defining_properties(cos):
    r, j = _pair(cos)
    u = oblique_covector(r, j.unsqueeze(0))
    assert torch.allclose(u @ r, torch.tensor(1.0), atol=1e-4), "u.r must be 1"
    assert abs((u @ j).item()) < 1e-4, "u.j must be 0"


@pytest.mark.parametrize("cos", [0.0, 0.18, 0.5])
def test_edit_removes_r_and_preserves_j(cos):
    """The point of the method, stated as an assertion."""
    r, j = _pair(cos)
    u = oblique_covector(r, j.unsqueeze(0))
    edit = lambda y: y - (y @ u) * r                 # w = 1, full removal

    # r is removed completely.
    assert edit(r).norm().item() < 1e-4
    # j passes through untouched.
    assert torch.allclose(edit(j), j, atol=1e-5)


def test_standard_projection_damages_j_when_correlated():
    """Why the method exists: the symmetric edit drags j down with r."""
    r, j = _pair(0.18)
    standard = j - (j @ r) * r                       # u = r
    oblique = j - (j @ oblique_covector(r, j.unsqueeze(0))) * r

    assert (standard - j).norm().item() > 0.15, "expected visible damage"
    assert (oblique - j).norm().item() < 1e-5, "oblique must be lossless on j"


def test_leverage_grows_as_directions_align():
    """The cost of the method is bounded by 1/sin(angle) and should be reported."""
    prev = 0.0
    for cos in (0.0, 0.18, 0.5, 0.9):
        r, j = _pair(cos)
        lev = leverage(r, oblique_covector(r, j.unsqueeze(0)))
        assert lev >= prev - 1e-6
        prev = lev
    # At the value measured on Qwen3-4B layer 21 the cost is negligible.
    r, j = _pair(0.18)
    assert leverage(r, oblique_covector(r, j.unsqueeze(0))) < 1.05


def test_parallel_directions_raise():
    """Inseparable behaviours must fail loudly, not silently do nothing."""
    r = F.normalize(torch.randn(32, generator=torch.Generator().manual_seed(1)), dim=0)
    with pytest.raises(ValueError, match="inseparable|no oblique projector"):
        oblique_covector(r, r.unsqueeze(0))


def test_subspace_preserves_every_direction():
    """rank>1 must zero the whole span, not just one member."""
    g = torch.Generator().manual_seed(2)
    r = F.normalize(torch.randn(48, generator=g), dim=0)
    J = F.normalize(torch.randn(3, 48, generator=g), dim=1)
    u = oblique_covector(r, J)
    assert torch.allclose(u @ r, torch.tensor(1.0), atol=1e-4)
    for row in J:
        assert abs((u @ row).item()) < 1e-4


def test_rank1_subspace_matches_single_direction():
    dirs = F.normalize(torch.randn(6, 32, generator=torch.Generator().manual_seed(3)), dim=1)
    B = preserve_subspace(dirs, 2.0, k=1)
    assert B.shape == (1, 32)
    assert torch.allclose(B[0], interpolate_direction(dirs, 2.0), atol=1e-5)


def test_interpolation_is_continuous():
    dirs = F.normalize(torch.randn(6, 32, generator=torch.Generator().manual_seed(4)), dim=1)
    a = interpolate_direction(dirs, 2.0)
    b = interpolate_direction(dirs, 2.999)
    c = interpolate_direction(dirs, 3.0)
    assert (b - c).norm() < (a - c).norm(), "index should move smoothly between layers"
