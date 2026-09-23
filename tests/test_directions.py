"""Direction extraction, separability reporting, and layer discovery.

None of this needs a real model: the functions take mean activations as plain
tensors, so the behaviour that matters can be asserted directly.
"""
import pytest
import torch
import torch.nn.functional as F

from obliteration.directions import (build_directions, find_decoder_layers,
                                     is_multimodal_config, separability)
from obliteration.metrics import kl_from_baseline
from obliteration.search import suggest_params


def _means(n_layers=8, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_layers + 1, hidden, generator=g),
            torch.randn(n_layers + 1, hidden, generator=g))


# --- directions ---


def test_directions_are_unit_length_per_layer():
    neutral, behaviour = _means()
    for proj in (True, False):
        d = build_directions(neutral, behaviour, project=proj)
        assert torch.allclose(d.norm(dim=1), torch.ones(d.shape[0]), atol=1e-5)


def test_projection_removes_the_neutral_component():
    """project=True must leave the direction orthogonal to neutral behaviour."""
    neutral, behaviour = _means()
    d = build_directions(neutral, behaviour, project=True)
    g = F.normalize(neutral, p=2, dim=1)
    assert (d * g).sum(dim=1).abs().max().item() < 1e-5


def test_projection_changes_the_result():
    neutral, behaviour = _means()
    assert not torch.allclose(build_directions(neutral, behaviour, project=True),
                              build_directions(neutral, behaviour, project=False))


def test_direction_sign_points_from_neutral_to_behaviour():
    """Sign matters: the edit subtracts along r, so a flipped r would add it."""
    hidden = 16
    neutral = torch.zeros(2, hidden)
    behaviour = torch.zeros(2, hidden)
    behaviour[:, 0] = 1.0
    d = build_directions(neutral, behaviour, project=False)
    assert d[0, 0] > 0.99


# --- separability ---


def test_separability_reports_per_layer_and_summary():
    a, b = _means(n_layers=10)
    d1 = build_directions(a, b, project=False)
    d2 = build_directions(b, a, project=False)
    s = separability(d1, d2)
    assert len(s["per_layer_abs_cos"]) == d1.shape[0]
    assert 0.0 <= s["median_abs_cos_upper_half"] <= 1.0
    assert s["max_abs_cos_upper_half"] >= s["median_abs_cos_upper_half"]


def test_separability_is_one_for_identical_directions():
    """The inseparable case must read as 1, since that is what gates the method."""
    a, b = _means()
    d = build_directions(a, b, project=False)
    s = separability(d, d)
    assert s["median_abs_cos_upper_half"] == pytest.approx(1.0, abs=1e-5)


def test_separability_is_zero_for_orthogonal_directions():
    hidden = 16
    d1 = torch.zeros(4, hidden); d1[:, 0] = 1.0
    d2 = torch.zeros(4, hidden); d2[:, 1] = 1.0
    assert separability(d1, d2)["median_abs_cos_upper_half"] == pytest.approx(0.0, abs=1e-6)


# --- layer discovery ---


def test_finds_layers_through_common_wrappers():
    for build in (
        lambda ls: type("M", (), {"model": type("I", (), {"layers": ls})()})(),
        lambda ls: type("M", (), {"language_model": type("I", (), {"layers": ls})()})(),
    ):
        layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
        assert find_decoder_layers(build(layers)) is layers


def test_missing_layers_raises_rather_than_returning_nothing():
    with pytest.raises(RuntimeError, match="could not locate decoder layers"):
        find_decoder_layers(type("M", (), {})())


# --- metrics ---


def test_kl_is_zero_against_itself():
    lp = F.log_softmax(torch.randn(8, 50), dim=-1)
    assert kl_from_baseline(lp, lp) == pytest.approx(0.0, abs=1e-6)


def test_kl_grows_with_divergence():
    g = torch.Generator().manual_seed(1)
    base = F.log_softmax(torch.randn(8, 50, generator=g), dim=-1)
    near = F.log_softmax(base.exp().log() + 0.01 * torch.randn(8, 50, generator=g), dim=-1)
    far = F.log_softmax(torch.randn(8, 50, generator=g), dim=-1)
    assert kl_from_baseline(near, base) < kl_from_baseline(far, base)


# --- search parameter sampling ---


class _FakeTrial:
    """Returns the midpoint of every suggested range, and records the bounds."""

    def __init__(self):
        self.bounds = {}

    def suggest_float(self, name, lo, hi):
        self.bounds[name] = (lo, hi)
        return (lo + hi) / 2


def test_suggested_params_stay_inside_the_intended_depth_band():
    """The bounds encode where refusal is mediated. Sampling the whole stack
    was measured to wreck the model, so they are part of the method."""
    t = _FakeTrial()
    n_layers = 36
    last = n_layers - 1.0
    di, params = suggest_params(t, n_layers)

    assert t.bounds["direction_index"] == (0.4 * last, 0.9 * last)
    assert 0.4 * last <= di <= 0.9 * last
    for comp in ("attn", "mlp"):
        assert t.bounds[f"{comp}.pos"] == (0.6 * last, 1.0 * last)
        assert t.bounds[f"{comp}.max_w"] == (0.8, 1.5)


def test_min_w_can_never_exceed_max_w():
    """Sampled as a fraction for exactly this reason: independent sampling
    inverted the triangle and made the edit stronger away from the peak."""
    for n_layers in (8, 36, 64):
        _, params = suggest_params(_FakeTrial(), n_layers)
        for comp, p in params.items():
            assert p["min_w"] <= p["max_w"], comp


def test_every_component_gets_a_full_kernel():
    _, params = suggest_params(_FakeTrial(), 24)
    for comp in ("attn", "mlp"):
        assert set(params[comp]) == {"max_w", "pos", "min_w", "dist"}


# --- model class selection ---


def test_multimodal_config_detected_by_nested_text_config():
    """Vision checkpoints nest vocab_size under text_config. Asking for the
    plain causal-LM class then fails with a bare "no attribute 'vocab_size'",
    which points nowhere near the real cause."""
    cfg = type("C", (), {"text_config": object(), "architectures": None})()
    assert is_multimodal_config(cfg)


def test_multimodal_config_detected_by_architecture_name():
    cfg = type("C", (), {"text_config": None,
                         "architectures": ["Qwen3_5ForConditionalGeneration"]})()
    assert is_multimodal_config(cfg)


def test_plain_causal_lm_config_is_not_multimodal():
    cfg = type("C", (), {"text_config": None,
                         "architectures": ["Qwen3ForCausalLM"]})()
    assert not is_multimodal_config(cfg)


def test_config_without_either_attribute_is_not_multimodal():
    assert not is_multimodal_config(type("C", (), {})())
