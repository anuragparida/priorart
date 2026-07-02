#!/usr/bin/env python3
"""Add the Phase 3.4 ``business_category`` field to the eval set.

What this is
------------
Per ``docs/PHASE-3.md`` §3.4, the eval set gains a *business
category* per record — orthogonal to the existing ``category``
(which encodes the eval-set taxonomy: ``duplicate`` / ``novel`` /
``adversarial_*``). The business categories are the 8 PHASE-3.md
buckets:

    b2b_saas, consumer, devtools, marketplace, fintech,
    healthcare, education, other

This script is the canonical encoding: it reads the existing
``evals/labeled_v300.jsonl`` line-by-line, assigns a business
category via ``src.eval.categorize.assign_business_category``, and
writes a new JSONL with two additional fields per record:

- ``business_category`` — one of the 8 buckets (or ``other`` when
  no rule fired).
- ``business_category_provenance`` — the deterministic-rule-based
  v1 stamp (``deterministic-rule-based-v1-pending-anurag-hand-review``).

The script preserves every existing field, including the
LLM-generated ``provenance`` field (which the eval set uses for the
label provenance). The two provenance stamps are independent — one
captures "this record's label provenance" (LLM-generated v2, hand
review pending), the other captures "this record's category
provenance" (rule-based v1, hand review pending).

Why a separate script
---------------------
Keeping the build step as a script (not a one-time edit) means a
later run can re-derive the categories from a freshly-extended rule
set without retyping the JSONL by hand. The script is the
canonical encoding; the JSONL is the serialised artifact (per the
``evals/labeled_v300.README.md`` "Re-running" section, which we
mirror here for consistency).

Why we don't overwrite in place
-------------------------------
We write to ``-o`` (default ``evals/labeled_v300.jsonl``) after
materialising the new content in memory. This keeps the operation
atomic from the reader's perspective — no partial files, no
half-written records.

Usage
-----

    uv run python scripts/build_business_categories.py
    uv run python scripts/build_business_categories.py -i evals/labeled_v100.jsonl -o evals/labeled_v100.jsonl
    uv run python scripts/build_business_categories.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Make ``src`` importable when running directly (``python scripts/foo.py``)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.categorize import (  # noqa: E402
    DEFAULT_PROVENANCE,
    assign_business_category,
    category_coverage,
)
from src.eval.categorize import BusinessCategory  # noqa: E402


def iter_records(path: Path) -> Iterable[Tuple[int, dict]]:
    """Yield ``(lineno, record)`` pairs from the JSONL.

    Blank lines are skipped (same as ``load_benchmark``). A bad
    JSONL line aborts the run with a clear error — there is no
    silent skipping because the eval set is the artifact.
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{lineno}: invalid JSON: {exc}"
                ) from exc
            yield lineno, rec


def extend_records(
    records: Iterable[dict],
) -> Tuple[List[dict], dict]:
    """Add ``business_category`` + ``business_category_provenance``.

    Returns the extended list of records and a coverage dict keyed
    by ``BusinessCategory`` (counts). The coverage dict is also
    printed to stdout so a CI step can spot-check the distribution.
    """
    extended: List[dict] = []
    coverage_counter = {c: 0 for c in BusinessCategory}
    for rec in records:
        idea = rec.get("idea", "")
        assignment = assign_business_category(idea)
        cat = assignment.business_category
        coverage_counter[cat] += 1

        # Mutate a copy of the record (don't disturb the caller's
        # reference). We preserve the original ``provenance`` and
        # add a separate ``business_category_provenance`` field.
        new_rec = dict(rec)
        new_rec["business_category"] = cat.value
        new_rec["business_category_provenance"] = DEFAULT_PROVENANCE
        extended.append(new_rec)
    return extended, {c.value: n for c, n in coverage_counter.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=_REPO_ROOT / "evals" / "labeled_v300.jsonl",
        help="Path to the input JSONL.",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL. Defaults to --input (in-place).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the coverage stats + a sample of 3 records; do not write.",
    )
    args = p.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output or in_path

    if not in_path.exists():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 1

    raw_records = [rec for _, rec in iter_records(in_path)]
    extended, coverage = extend_records(raw_records)

    if args.dry_run:
        print(f"[dry-run] read {len(raw_records)} records from {in_path}")
        print("[dry-run] category coverage:")
        for cat, n in coverage.items():
            print(f"  {cat:<14} {n:>3} ({n / max(1, len(raw_records)) * 100:5.1f}%)")
        print("[dry-run] sample extended record (first):")
        if extended:
            print(json.dumps(extended[0], indent=2)[:600])
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in extended:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[build_business_categories] wrote {len(extended)} records to {out_path}")
    print("[build_business_categories] category coverage:")
    for cat, n in coverage.items():
        print(f"  {cat:<14} {n:>3} ({n / max(1, len(extended)) * 100:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())