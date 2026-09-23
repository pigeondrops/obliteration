"""Prompt set integrity.

The grounding prompts contain a blank line between context and question. Writing
them one-per-line and reading back with splitlines() silently turns each prompt
into two fragments: a context with no question, and a question with no context.
A bare question is not answerable either way, so the direction that comes out
measures question topic rather than groundedness. These tests exist so that
cannot happen again unnoticed.
"""
import json

import pytest

from obliteration.data import (grounding_pairs, load_prompts, match_sizes,
                               write_prompts)


def test_pairs_are_matched_and_aligned():
    ans, unans = grounding_pairs(64)
    assert len(ans) == len(unans) == 64
    for a, u in zip(ans, unans):
        ctx_a = a.split("Question:")[0]
        ctx_u = u.split("Question:")[0]
        assert ctx_a == ctx_u, "the two sides must share a byte-identical context"
        assert a != u, "they must differ in the question"


def test_unanswerable_question_is_absent_from_context():
    """The whole contrast depends on this actually holding."""
    ans, unans = grounding_pairs(64)
    for a, u in zip(ans, unans):
        ctx = a.split("Question:")[0]
        # The answerable question asks about a fact stated in the context; the
        # unanswerable one asks about the single fact that was withheld.
        assert "Question:" in u
        q_u = u.split("Question:")[1].strip()
        assert q_u not in ctx


def test_prompts_contain_newlines():
    """Guards the premise of the round-trip tests below."""
    ans, _ = grounding_pairs(4)
    assert any("\n" in p for p in ans)


def test_roundtrip_preserves_prompt_count(tmp_path):
    ans, _ = grounding_pairs(32)
    path = write_prompts(tmp_path / "a.jsonl", ans)
    back = load_prompts(path)
    assert len(back) == 32, "multi-line prompts must not be split on read"
    assert back == ans


def test_write_upgrades_txt_to_jsonl_for_multiline(tmp_path):
    ans, _ = grounding_pairs(8)
    path = write_prompts(tmp_path / "a.txt", ans)
    assert path.suffix == ".jsonl", "multi-line sets must not be written as .txt"
    assert load_prompts(path) == ans


def test_multiline_txt_is_rejected_not_silently_split(tmp_path):
    """Reading it one-per-line would yield twice as many prompts, each half a
    record, so it must be refused instead."""
    bad = tmp_path / "bad.txt"
    ans, _ = grounding_pairs(8)
    bad.write_text("\n".join(ans) + "\n")          # the wrong way to write these
    with pytest.raises(SystemExit, match="blank lines|jsonl"):
        load_prompts(bad)


def test_single_line_txt_still_works(tmp_path):
    """Refusal sets are one sentence per line and must stay simple."""
    p = tmp_path / "harmful.txt"
    p.write_text("first prompt\nsecond prompt\n\n")
    assert load_prompts(p) == ["first prompt", "second prompt"]


def test_incidental_blank_lines_are_tolerated(tmp_path):
    """Real prompt lists contain the odd stray blank. That must not be fatal.

    Refusal sets assembled from public corpora routinely carry a handful of blank
    lines. The guard below must not reject those.
    """
    p = tmp_path / "harmless.txt"
    lines = [f"prompt {i}" for i in range(200)]
    lines.insert(20, "")
    lines.insert(120, "")
    p.write_text("\n".join(lines) + "\n")
    assert len(load_prompts(p)) == 200


def test_jsonl_is_valid_json_per_line(tmp_path):
    ans, _ = grounding_pairs(5)
    path = write_prompts(tmp_path / "a.jsonl", ans)
    for line in path.read_text().splitlines():
        assert isinstance(json.loads(line), str)


def test_missing_file_fails_with_a_useful_message(tmp_path):
    with pytest.raises(SystemExit, match="missing prompt file"):
        load_prompts(tmp_path / "nope.txt")


def test_match_sizes_truncates_both_sides():
    a, b, k = match_sizes(list(range(10)), list(range(4)), 8)
    assert k == 4 and len(a) == len(b) == 4


def test_shipped_data_roundtrips():
    """The files committed to the repo must themselves be readable."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("answerable.jsonl", "unanswerable.jsonl"):
        f = root / "data" / name
        if f.exists():
            assert len(load_prompts(f)) == 256
