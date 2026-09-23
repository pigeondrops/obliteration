"""Applying the edit across a layer stack, and the artifacts written beside it.

`apply_to_weights` walks the stack and decides which matrices get edited, at
what strength, and which get skipped. Those decisions are what make an export
either a correctly edited model or a quietly unmodified copy, so they are
asserted here against fake layers rather than left to a real run.
"""
import json

import pytest
import torch
import torch.nn.functional as F

from obliteration.export import apply_to_weights, write_manifest


class _Sub(torch.nn.Module):
    """One writing sublayer, with the leaf name the export looks for."""

    def __init__(self, hidden, leaf):
        super().__init__()
        setattr(self, leaf, torch.nn.Linear(hidden, hidden, bias=False))


class _Layer(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Sub(hidden, "o_proj")
        self.mlp = _Sub(hidden, "down_proj")


def _stack(n=6, hidden=32):
    return torch.nn.ModuleList([_Layer(hidden) for _ in range(n)])


def _dirs(n=6, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n + 1, hidden, generator=g), dim=1)


def _best(pos=2.0, dist=10.0, max_w=1.0):
    p = {"max_w": max_w, "pos": pos, "min_w": max_w, "dist": dist}
    return {"direction_index": 2.0, "params": {"attn": dict(p), "mlp": dict(p)},
            "refusals": 0, "kl": 0.01}


def test_edits_every_writing_matrix_in_range():
    layers = _stack(6)
    edited, skipped, lev = apply_to_weights(layers, _dirs(6), _best(), log=lambda *a: None)
    assert edited == 12, "two writing matrices per layer, six layers"
    assert skipped == 0
    assert lev == pytest.approx(1.0), "leverage is 1 without preservation"


def test_layers_outside_the_kernel_are_left_untouched():
    """The triangular kernel returning None must mean no write at all."""
    layers = _stack(6)
    before = [l.self_attn.o_proj.weight.clone() for l in layers]
    apply_to_weights(layers, _dirs(6), _best(pos=0.0, dist=1.5), log=lambda *a: None)
    after = [l.self_attn.o_proj.weight for l in layers]
    assert not torch.allclose(before[0], after[0]), "layer 0 is at the peak"
    assert torch.allclose(before[5], after[5]), "layer 5 is outside dist"


def test_weights_actually_change():
    layers = _stack(3)
    before = layers[1].mlp.down_proj.weight.clone()
    apply_to_weights(layers, _dirs(3), _best(), log=lambda *a: None)
    assert not torch.allclose(before, layers[1].mlp.down_proj.weight)


def test_preservation_changes_the_edit_and_reports_leverage():
    """With a preserved direction the covector differs from r, so both the
    resulting weights and the reported leverage must differ."""
    dirs, keep = _dirs(4, seed=0), _dirs(4, seed=1)

    _, _, lev_plain = apply_to_weights(_stack(4), dirs, _best(), log=lambda *a: None)
    _, _, lev_obl = apply_to_weights(_stack(4), dirs, _best(), preserve_dirs=keep,
                                     log=lambda *a: None)
    assert lev_plain == pytest.approx(1.0)
    assert lev_obl > 1.0, "oblique edit is amplified"


def test_inseparable_preservation_falls_back_instead_of_crashing():
    """If refusal lies inside the preserved span there is no projector. The
    export must still produce a model, with a warning, not raise."""
    dirs = _dirs(4)
    msgs = []
    edited, _, lev = apply_to_weights(_stack(4), dirs, _best(), preserve_dirs=dirs,
                                      log=msgs.append)
    assert edited > 0
    assert lev == pytest.approx(1.0), "fell back to the standard edit"
    assert any("inseparable" in m for m in msgs)


def test_zero_edits_is_detectable_by_the_caller():
    """A stack whose names never match must report zero, so the CLI can refuse
    to write what would be an unmodified copy."""
    class _Foreign(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Linear(32, 32, bias=False)

    layers = torch.nn.ModuleList([_Foreign() for _ in range(3)])
    edited, _, _ = apply_to_weights(layers, _dirs(3), _best(), log=lambda *a: None)
    assert edited == 0


# --- manifest ---


def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    best = _best()
    write_manifest(tmp_path, "Qwen/Qwen3-4B", best, True, 12, 1.017)
    m = json.loads((tmp_path / "abliteration.json").read_text())
    for key in ("method", "model", "preserve_grounding", "direction_index",
                "params", "refusals_after", "kl", "matrices_edited",
                "oblique_leverage"):
        assert key in m, key
    assert m["preserve_grounding"] is True
    assert m["oblique_leverage"] == 1.017


def test_manifest_marks_a_plain_abliteration_as_such(tmp_path):
    write_manifest(tmp_path, "m", _best(), False, 4, 1.0)
    assert json.loads((tmp_path / "abliteration.json").read_text())["preserve_grounding"] is False
