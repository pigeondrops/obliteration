"""Prompt sets.

Four sets are needed, in two matched pairs:

    harmful / harmless          -> the refusal direction
    unanswerable / answerable   -> the direction to preserve

Matching is not a detail. Everything the two sides of a pair share cancels in
the difference of means, so the direction isolates the behaviour rather than
topic, length or dataset identity. The grounding pairs below are matched to the
byte: both sides carry the same three-fact context about the same invented
entity, and differ only in whether the queried fact is present.

The refusal pair is not vendored here. scripts/fetch_data.py fetches it from
upstream; see the "Data" section of the README.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Invented vocabulary. Nonsense on purpose: a real entity would let the model
# answer from parametric memory instead of from the context, which would measure
# recall rather than grounding.
_A = ["Ki", "Ka", "Nu", "Do", "Li", "Va", "Tu", "Ba", "Ze", "Mo", "Ra", "Su"]
_B = ["na", "do", "ge", "vo", "za", "mi", "lu", "te", "ro", "ka"]
_C = ["guok", "eth", "pei", "arn", "usk", "ilo", "adem", "yrn", "osh", "ekt"]
_EP = ["the Pale", "the Younger", "the Grey", "the Third", "the Quiet",
       "the Elder", "the Bright", "the Far", "the Silent", "the Red"]
_DAYS = ["Lowmoon", "Highmoon", "Firstwater", "Ashday", "Tallnight", "Greenturn"]
_FACTS = [
    ("export", "Their main export is {v} ore.", "What is the main export of the {e}?"),
    ("ruler", "They are ruled by {v}.", "Who rules the {e}?"),
    ("food", "Their staple food is {v} root.", "What is the staple food of the {e}?"),
    ("festival", "They hold their festival on {v}.", "On what day do the {e} hold their festival?"),
    ("homeworld", "Their homeworld is {v}.", "What is the homeworld of the {e}?"),
    ("lifespan", "Their typical lifespan is {v} years.", "What is the typical lifespan of the {e}?"),
]


def _name(rng):
    return rng.choice(_A) + rng.choice(_B) + rng.choice(_C)


def _value(rng, key):
    if key == "ruler":
        return f"{_name(rng)} {rng.choice(_EP)}"
    if key == "festival":
        return rng.choice(_DAYS)
    if key == "lifespan":
        return str(rng.randint(40, 900))
    return _name(rng)[:5]


def grounding_pairs(n: int = 256, seed: int = 0):
    """-> (answerable, unanswerable), equal length and index aligned.

    Each pair shares a byte-identical context and differs only in whether the
    question is answerable from it. The unanswerable side is what a
    well-behaved model should hedge on.
    """
    rng = random.Random(seed)
    ans, unans = [], []
    for _ in range(n):
        entity = _name(rng)
        keys = rng.sample(range(len(_FACTS)), 4)
        shown, hidden = keys[:3], keys[3]
        lines = [_FACTS[k][1].format(v=_value(rng, _FACTS[k][0])) for k in shown]
        rng.shuffle(lines)
        ctx = "Context: " + " ".join(lines) + "\n\n"
        ans.append(ctx + "Question: " + _FACTS[rng.choice(shown)][2].format(e=entity))
        unans.append(ctx + "Question: " + _FACTS[hidden][2].format(e=entity))
    return ans, unans


def load_prompts(path) -> list:
    """Read a prompt file.

    Two formats, chosen by extension:

      .txt    one prompt per line. Fine for the refusal sets, which are single
              sentences.
      .jsonl  one JSON string per line. Required for any prompt containing a
              newline.

    The grounding prompts embed a blank line between context and question, so
    splitting a .txt on newlines would turn each one into a context-only row and
    a question-only row. That is not a cosmetic difference: a bare question with
    no context is not answerable either way, so the resulting direction measures
    question topic rather than groundedness. Hence .jsonl for those.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"missing prompt file: {p}\n"
            "Run scripts/fetch_data.py, or point the flag at your own file.")

    if p.suffix == ".jsonl":
        out = []
        for i, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{p}:{i} is not valid JSON: {e}")
            # A prompt must be a string. A bare number or object parses fine and
            # then fails much later inside the tokenizer with an opaque error.
            if not isinstance(value, str):
                raise SystemExit(
                    f"{p}:{i} holds {type(value).__name__}, not a string. "
                    "Each line must be one JSON-encoded prompt.")
            out.append(value)
        return _non_empty(out, p)

    # strip("\n") first: a trailing newline is universal and carries no meaning,
    # but on a short file it would dominate the ratio below.
    lines = p.read_text().strip("\n").splitlines()
    prompts = [l for l in lines if l.strip()]
    blanks = len(lines) - len(prompts)

    # A .txt is one prompt per line and stray blanks are dropped. But a file of
    # MULTI-line records written this way is a silent corruption, so catch the
    # signature: record separators recur often, incidental blanks do not. The
    # grounding format (context, blank, question) is one blank in three; a real
    # prompt list is well under one percent.
    if lines and blanks / max(len(lines), 1) > 0.25:
        raise SystemExit(
            f"{p} is {100 * blanks // len(lines)}% blank lines, which means it "
            "holds multi-line prompts that cannot be read one-per-line. Each "
            "record would be split into fragments. Save it as .jsonl instead "
            "(one JSON string per line).")
    return _non_empty(prompts, p)


def _non_empty(prompts, path):
    """An empty prompt set is never what the caller wanted.

    Returning [] silently pushes the failure into residual_means, which then
    reports "every prompt produced non-finite activations" and sends the user
    looking for a numerical problem that does not exist.
    """
    if not prompts:
        raise SystemExit(f"{path} contains no prompts")
    return prompts


def write_prompts(path, prompts) -> Path:
    """Write a prompt list, picking a format that round-trips.

    Multi-line prompts go to .jsonl. Round-tripping is checked, not assumed:
    writing a set that cannot be read back is a silent corruption of the whole
    experiment.
    """
    p = Path(path)
    multiline = any("\n" in s for s in prompts)
    if multiline and p.suffix != ".jsonl":
        p = p.with_suffix(".jsonl")
    if p.suffix == ".jsonl":
        p.write_text("".join(json.dumps(s) + "\n" for s in prompts))
    else:
        p.write_text("\n".join(prompts) + "\n")

    back = load_prompts(p)
    if back != list(prompts):
        raise RuntimeError(
            f"{p} did not round-trip: wrote {len(prompts)} prompts, read back "
            f"{len(back)}")
    return p


# Kept as an alias: the refusal sets are single-line and this reads better there.
load_lines = load_prompts


def match_sizes(a: list, b: list, n: int):
    """Truncate both sides to the same length.

    Mismatched sizes make the difference of means capture dataset identity
    rather than behaviour. Measured with an unmatched pair, that was roughly a
    35 degree rotation away from the true refusal direction, which is enough to
    make the whole edit land in the wrong place.
    """
    k = min(n, len(a), len(b))
    if k == 0:
        raise SystemExit("one of the prompt sets is empty, so no direction can "
                         "be fitted")
    if k < n:
        print(f"note: only {k} matched pairs available (asked for {n})")
    return a[:k], b[:k], k


def split_fit_eval(prompts, n_fit, n_eval):
    """Disjoint fit and eval slices.

    Taking prompts[:n_fit] and prompts[-n_eval:] independently overlaps as soon
    as the set is smaller than n_fit + n_eval, which fits the direction on the
    same prompts used to score it. The eval slice is reserved first so the two
    can never intersect.
    """
    if len(prompts) < n_eval + 1:
        raise SystemExit(f"need at least {n_eval + 1} prompts to hold out "
                         f"{n_eval} for evaluation, got {len(prompts)}")
    ev = prompts[-n_eval:]
    rest = prompts[:-n_eval]
    if len(rest) < n_fit:
        print(f"note: only {len(rest)} prompts left to fit on after holding out "
              f"{n_eval} for evaluation (asked for {n_fit})")
    return rest[:n_fit], ev
