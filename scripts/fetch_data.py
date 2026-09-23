#!/usr/bin/env python3
"""Fetch the refusal prompt sets from their original sources.

Neither refusal set is vendored. Both are fetched from their upstream
repositories at the version those currently serve, so the licences stay with
their owners and this repo carries no stale copy.

Alpaca additionally cannot be vendored: the repository is Apache-2.0 but the
dataset itself is CC BY-NC 4.0, which is not compatible with redistributing it
inside an MIT-licensed repo.

The grounding pairs are generated locally and need no download. See
obliteration/data.py.

Usage:
    python scripts/fetch_data.py            # writes data/harmful.txt, data/harmless.txt
    python scripts/fetch_data.py --grounding-only
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

# Run straight from a clone without installing first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obliteration.data import grounding_pairs, write_prompts

# AdvBench, from "Universal and Transferable Adversarial Attacks on Aligned
# Language Models", Zou, Wang, Carlini et al., arXiv:2307.15043. MIT.
ADVBENCH = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
            "data/advbench/harmful_behaviors.csv")
# Stanford Alpaca, Taori, Gulrajani, Zhang et al. Code is Apache-2.0, the
# DATASET is CC BY-NC 4.0: non-commercial, attribution required.
ALPACA = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/"
          "alpaca_data.json")


def fetch(url: str) -> bytes:
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=520)
    ap.add_argument("--grounding-only", action="store_true")
    ap.add_argument("--grounding-n", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Grounding pairs: local, synthetic, always safe to write.
    ans, unans = grounding_pairs(args.grounding_n)
    # .jsonl, not .txt: these prompts contain a blank line between context
    # and question, so one-per-line would split each into two fragments.
    a = write_prompts(out / "answerable.jsonl", ans)
    u = write_prompts(out / "unanswerable.jsonl", unans)
    print(f"wrote {len(ans)} answerable / {len(unans)} unanswerable (round-trip checked)")

    if args.grounding_only:
        return 0

    try:
        rows = list(csv.DictReader(io.StringIO(fetch(ADVBENCH).decode())))
        harmful = [r["goal"].strip() for r in rows if r.get("goal", "").strip()]
        (out / "harmful.txt").write_text("\n".join(harmful[:args.n]) + "\n")
        print(f"wrote {min(len(harmful), args.n)} harmful prompts")
    except Exception as e:
        print(f"could not fetch AdvBench ({e}); supply --harmful yourself")

    try:
        data = json.loads(fetch(ALPACA).decode())
        # No-input instructions only, so the prompt shape matches AdvBench's.
        harmless = [d["instruction"].strip() for d in data
                    if not d.get("input", "").strip()]
        (out / "harmless.txt").write_text("\n".join(harmless[:args.n]) + "\n")
        print(f"wrote {min(len(harmless), args.n)} harmless prompts")
    except Exception as e:
        print(f"could not fetch Alpaca ({e}); supply --harmless yourself")

    print("\nKeep the two sets the same size. Unmatched sets make the difference "
          "of means capture dataset identity rather than refusal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
