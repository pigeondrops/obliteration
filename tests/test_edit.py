"""The edit itself: the depth kernel, the hook, and the weight write.

`_project_` is the function that actually modifies the checkpoint, and the hook
is what the search scores. If these two disagree the search optimises one thing
and the export ships another, so the equivalence is asserted directly.
"""
import torch
import torch.nn.functional as F

from obliteration.ablator import COMPONENTS, ObliqueAblator, layer_weight
from obliteration.export import _project_


def _vecs(hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = F.normalize(torch.randn(hidden, generator=g), dim=0)
    u = F.normalize(torch.randn(hidden, generator=g), dim=0)
    return r, u


# --- triangular depth kernel ---


def test_kernel_peaks_at_pos_and_decays():
    p = {"max_w": 1.2, "pos": 10.0, "min_w": 0.2, "dist": 5.0}
    assert layer_weight(10, p) == 1.2
    assert layer_weight(12, p) < layer_weight(11, p) < 1.2
    assert layer_weight(8, p) == layer_weight(12, p), "kernel must be symmetric"


def test_kernel_returns_none_outside_dist():
    """None means the layer is skipped entirely, not edited at strength zero."""
    p = {"max_w": 1.0, "pos": 10.0, "min_w": 0.0, "dist": 3.0}
    assert layer_weight(13, p) is not None
    assert layer_weight(14, p) is None
    assert layer_weight(6, p) is None


def test_kernel_never_exceeds_max_w():
    p = {"max_w": 0.9, "pos": 4.0, "min_w": 0.1, "dist": 6.0}
    ws = [layer_weight(i, p) for i in range(12)]
    assert max(w for w in ws if w is not None) <= 0.9 + 1e-9


# --- the weight write ---


def test_project_2d_matches_the_definition():
    r, u = _vecs()
    W = torch.randn(32, 16)
    got = _project_(W.clone(), r, u, 0.7)
    want = W - 0.7 * torch.outer(r, u @ W)
    assert torch.allclose(got, want, atol=1e-6)


def test_project_handles_both_fused_expert_layouts():
    """A fused MoE stores experts as one 3-D tensor, in either index order."""
    r, u = _vecs()
    for shape in [(4, 32, 16), (4, 16, 32)]:
        W = torch.randn(*shape)
        got = _project_(W.clone(), r, u, 0.5)
        assert got is not None, f"layout {shape} was not recognised"
        assert got.shape == W.shape
        assert not torch.allclose(got, W), "tensor was returned unmodified"


def test_project_returns_none_when_no_axis_matches():
    """Callers count this as skipped. Silently guessing an axis would corrupt."""
    r, u = _vecs()
    assert _project_(torch.randn(8, 8), r, u, 1.0) is None


def test_project_preserves_dtype():
    r, u = _vecs()
    W = torch.randn(32, 16, dtype=torch.bfloat16)
    assert _project_(W.clone(), r, u, 0.5).dtype == torch.bfloat16


def test_project_removes_r_from_every_column():
    """With u = r and w = 1 the refusal component should be gone."""
    r, _ = _vecs()
    W = torch.randn(32, 16)
    out = _project_(W.clone(), r, r, 1.0)
    assert (r @ out).abs().max().item() < 1e-5


# --- hook and weight write must agree ---


class _Block(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class _Layer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Block(hidden)
        self.mlp = _Block(hidden)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


def test_hook_equals_editing_the_weight():
    """(W - w r u^T W) x  ==  y - w (u.y) r.

    The search relies on this. If it ever stops holding, every trial score is
    measuring a different model from the one that gets exported.
    """
    hidden = 32
    torch.manual_seed(0)
    layer = _Layer(hidden)
    r, u = _vecs(hidden)
    x = torch.randn(4, hidden)

    params = {"attn": {"max_w": 1.0, "pos": 0.0, "min_w": 1.0, "dist": 1.0},
              "mlp": {"max_w": 0.0, "pos": 0.0, "min_w": 0.0, "dist": 1.0}}
    dirs = torch.stack([r, r, r])

    abl = ObliqueAblator(None, [layer]).install()
    abl.configure(dirs, 0.0, params, preserve_dirs=None)
    hooked = layer.self_attn(x)
    abl.remove()

    W = layer.self_attn.o_proj.weight.data
    layer.self_attn.o_proj.weight.data = _project_(W.clone(), r, r, 1.0)
    edited = layer.self_attn(x)

    assert torch.allclose(hooked, edited, atol=1e-5)


def test_ablator_with_no_preserve_dirs_is_standard_abliteration():
    """u must fall back to r, so the class doubles as the baseline."""
    hidden = 16
    layer = _Layer(hidden)
    r = F.normalize(torch.randn(hidden, generator=torch.Generator().manual_seed(3)), dim=0)
    dirs = torch.stack([r, r])
    abl = ObliqueAblator(None, [layer])
    abl.configure(dirs, 0.0, {"attn": {"max_w": 1.0, "pos": 0.0, "min_w": 1.0, "dist": 1.0},
                              "mlp": {"max_w": 1.0, "pos": 0.0, "min_w": 1.0, "dist": 1.0}},
                  preserve_dirs=None)
    for (rr, uu, _w) in abl._cfg.values():
        assert torch.allclose(rr, uu), "without preservation u should equal r"


def test_ablator_records_inseparable_layers_instead_of_failing():
    """A layer where the behaviours are parallel falls back and is reported."""
    hidden = 16
    layer = _Layer(hidden)
    r = F.normalize(torch.randn(hidden, generator=torch.Generator().manual_seed(5)), dim=0)
    dirs = torch.stack([r, r])
    abl = ObliqueAblator(None, [layer])
    abl.configure(dirs, 0.0, {"attn": {"max_w": 1.0, "pos": 0.0, "min_w": 1.0, "dist": 1.0},
                              "mlp": {"max_w": 1.0, "pos": 0.0, "min_w": 1.0, "dist": 1.0}},
                  preserve_dirs=dirs)          # preserve == refusal: inseparable
    assert abl.inseparable_layers == [0]


def test_components_cover_both_writing_sublayers():
    assert set(COMPONENTS) == {"attn", "mlp"}
