"""Malformed input must fail at the point it is read.

A prompt set that is empty, wrongly formatted, or degenerate should stop the run
immediately with a message naming the file. Left unchecked, bad input survives
far enough to surface as an opaque crash deep in the pipeline, or to produce a
plausible-looking result that is quietly wrong.
"""
import pytest
import torch

from obliteration.data import (load_prompts, match_sizes, split_fit_eval,
                               write_prompts)
from obliteration.projector import oblique_covector


# --- prompt files that contain nothing usable ---


def test_empty_file_is_rejected(tmp_path):
    """An empty set must be caught here, not inside residual_means, which can
    only report "every prompt produced non-finite activations" and points at a
    numerical problem that does not exist."""
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(SystemExit, match="no prompts"):
        load_prompts(p)


def test_file_of_only_blank_lines_is_rejected(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("\n\n\n\n")
    with pytest.raises(SystemExit, match="no prompts"):
        load_prompts(p)


def test_empty_jsonl_is_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="no prompts"):
        load_prompts(p)


# --- jsonl payloads that are not prompts ---


def test_non_string_jsonl_entry_is_rejected(tmp_path):
    """A bare number parses as JSON and then fails inside the tokenizer with an
    error that says nothing about the data file."""
    p = tmp_path / "n.jsonl"
    p.write_text('"ok"\n123\n')
    with pytest.raises(SystemExit, match="not a string"):
        load_prompts(p)


def test_object_jsonl_entry_is_rejected(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text('"ok"\n{"prompt": "x"}\n')
    with pytest.raises(SystemExit, match="not a string"):
        load_prompts(p)


def test_malformed_jsonl_reports_the_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('"ok"\nnot json\n')
    with pytest.raises(SystemExit, match=r"bad\.jsonl:2"):
        load_prompts(p)


# --- non-finite activations ---


def test_non_finite_refusal_direction_raises_clearly():
    """Without this check a non-finite direction reaches linalg.lstsq, which
    fails with an INTERNAL ASSERT that reads like a torch bug. It would also
    slip past the denominator guard, since NaN < 1e-6 is False, leaving an
    all-NaN covector to be written into the weights."""
    r = torch.full((8,), float("nan"))
    j = torch.randn(8)
    with pytest.raises(ValueError, match="non-finite"):
        oblique_covector(r, j.unsqueeze(0))


def test_non_finite_preserved_direction_raises_clearly():
    r = torch.randn(8)
    j = torch.full((8,), float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        oblique_covector(r, j.unsqueeze(0))


# --- fit and eval must not share prompts ---


def test_fit_and_eval_never_overlap():
    """Slicing [:n] and [-n_eval:] independently overlaps as soon as the set is
    smaller than n + n_eval, fitting the direction on the prompts used to
    score it."""
    for total in (520, 200, 100, 60):
        prompts = list(range(total))
        fit, ev = split_fit_eval(prompts, 256, 48)
        assert not (set(fit) & set(ev)), f"overlap at total={total}"
        assert len(ev) == 48


def test_split_keeps_the_full_fit_set_when_there_is_room():
    fit, ev = split_fit_eval(list(range(520)), 256, 48)
    assert len(fit) == 256 and len(ev) == 48


def test_split_shrinks_the_fit_set_rather_than_stealing_from_eval():
    fit, ev = split_fit_eval(list(range(100)), 256, 48)
    assert len(ev) == 48, "eval size is the fixed commitment"
    assert len(fit) == 52


def test_split_rejects_a_set_too_small_to_hold_out():
    with pytest.raises(SystemExit, match="at least"):
        split_fit_eval(list(range(10)), 8, 48)


# --- degenerate matched sets ---


def test_match_sizes_rejects_an_empty_side():
    """k=0 must stop the run rather than yield empty fit sets."""
    with pytest.raises(SystemExit, match="empty"):
        match_sizes([1, 2, 3], [], 10)


# --- round-trip helper contract ---


def test_write_prompts_returns_the_path_it_used(tmp_path):
    """It may rewrite the suffix, so the caller needs the real path back."""
    out = write_prompts(tmp_path / "a.txt", ["one", "two"])
    assert out.exists()
    assert load_prompts(out) == ["one", "two"]
