"""Render a leaderboard diff for the PR comment — Phase 3.6 (card t_e0f62c2a).

What this is
------------
Compares two leaderboard CSVs (the one we just produced vs the
one committed on the base branch) and renders a Markdown table
suitable for posting as a PR comment. The diff is "best row per
config" — the row where ``selected_threshold=True`` — because
that's the row the regression gate inspects, and that's the row
a PR reviewer cares about.

CLI
---
::

    # In a PR context (default workflow behaviour):
    git show origin/$BASE_BRANCH:results/leaderboard.csv > /tmp/base.csv
    uv run python scripts/ci/leaderboard_diff.py \\
        --base /tmp/base.csv \\
        --head results/leaderboard.csv \\
        --output /tmp/leaderboard_diff.md

    # Print to stdout:
    uv run python scripts/ci/leaderboard_diff.py \\
        --base /tmp/base.csv --head results/leaderboard.csv

Exit codes:

* 0  — diff rendered (whether or not numbers changed).
* 1  — base or head CSV missing / unreadable.
* 2  — head CSV has no selected rows for any of the 3 configs
        (sweep didn't write a usable leaderboard; should have
        been caught by eval_gate earlier).

Pure
----
Every rendering function is a pure function over the parsed
rows. Unit tests pass synthetic ``list[dict]`` in.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Reuse the gate's row-coercion helpers. Importing the module
# (not duplicating the coercion) is the contract: the gate and
# the diff agree on what "selected" and "mrr" mean.
from scripts.ci.eval_gate import (  # noqa: E402
    SELECTED_THRESHOLD_SENTINEL,  # noqa: F401 — re-exported for tests
    _coerce_float,
    find_selected_row,
    read_leaderboard_csv,
)

# The 3 sweep configs in display order. The diff lists all 3 so
# a reviewer can see "I changed dense_bge_m3, here's the
# unchanged bm25 / hybrid_rrf" at a glance.
DISPLAY_CONFIGS: Sequence[str] = (
    "dense_bge_m3",
    "bm25",
    "hybrid_rrf",
)

# Columns shown in the diff table. Kept tight (8 cols) so the
# Markdown fits inside a PR-comment width on mobile. ECE and
# novel_set_mrr are intentionally excluded — the headline
# reviewer signal is MRR / FPR / threshold, and the deeper
# diagnostic lives in the dashboard.
DIFF_COLUMNS: Sequence[tuple[str, str]] = (
    ("config", "config"),
    ("threshold", "threshold"),
    ("mrr", "MRR"),
    ("fpr_on_novel", "FPR-on-novel"),
    ("ndcg_at_10", "nDCG@10"),
    ("precision_at_5", "P@5"),
    ("recall_at_10", "R@10"),
    ("ece", "ECE"),
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BestRow:
    """The selected row for one config, projected to the diff schema."""

    config: str
    threshold: float
    mrr: float
    fpr_on_novel: float
    ndcg_at_10: float
    precision_at_5: float
    recall_at_10: float
    ece: float

    @classmethod
    def from_row(cls, row: dict[str, str]) -> BestRow:
        return cls(
            config=row.get("config", ""),
            threshold=_coerce_float(row.get("threshold", "0")),
            mrr=_coerce_float(row.get("mrr", "0")),
            fpr_on_novel=_coerce_float(row.get("fpr_on_novel", "0")),
            ndcg_at_10=_coerce_float(row.get("ndcg_at_10", "0")),
            precision_at_5=_coerce_float(row.get("precision_at_5", "0")),
            recall_at_10=_coerce_float(row.get("recall_at_10", "0")),
            ece=_coerce_float(row.get("ece", "0")),
        )


def collect_best_rows(
    rows: Iterable[dict[str, str]],
    *,
    configs: Sequence[str] = DISPLAY_CONFIGS,
) -> dict[str, BestRow | None]:
    """For each config, return the selected row (or None if missing).

    Pure function over the parsed rows. Used by ``render_diff``
    and by unit tests that build synthetic leaderboard CSVs.
    """
    out: dict[str, BestRow | None] = {}
    for cfg in configs:
        sel = find_selected_row(list(rows), config=cfg)
        out[cfg] = BestRow.from_row(sel) if sel is not None else None
    return out


def _fmt_num(value: float) -> str:
    """Format a numeric cell to 3 decimals, trimming trailing zeros.

    Mirrors the convention in the README's leaderboard screenshot
    (see ``scripts/render_leaderboard_v2_screenshot.py``) so the
    Markdown diff and the PNG look the same.
    """
    s = f"{value:.3f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def render_diff(
    base_rows: Sequence[dict[str, str]],
    head_rows: Sequence[dict[str, str]],
    *,
    configs: Sequence[str] = DISPLAY_CONFIGS,
    gate_thresholds: dict[str, float] | None = None,
) -> str:
    """Render the diff as a Markdown block.

    Pure — no I/O, no DB. ``gate_thresholds`` is optional; when
    provided the per-row cells are coloured relative to the gate
    floors so a reviewer can spot a regression at a glance.
    """
    base = collect_best_rows(base_rows, configs=configs)
    head = collect_best_rows(head_rows, configs=configs)

    # Header row.
    headers = [label for _, label in DIFF_COLUMNS]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"

    # Body rows: one per config, base + head side-by-side when both exist.
    body: list[str] = []
    for cfg in configs:
        b = base.get(cfg)
        h = head.get(cfg)
        if b is None and h is None:
            body.append(
                "| "
                + " | ".join([cfg, "—", "—", "—", "—", "—", "—", "—"])
                + " |"
            )
            continue
        # Pick the head row's values if present, else base — this
        # is the "what does the world look like now" row, with
        # the base values shown in the column comment.
        ref = h if h is not None else b
        assert ref is not None
        cells = [ref.config, _fmt_num(ref.threshold)]
        for key, _label in DIFF_COLUMNS[2:]:
            head_val = getattr(h, key) if h is not None else None
            base_val = getattr(b, key) if b is not None else None
            if head_val is None and base_val is not None:
                cells.append(f"{_fmt_num(base_val)} (base only)")
            elif base_val is None and head_val is not None:
                cells.append(f"{_fmt_num(head_val)} (new)")
            elif head_val is None and base_val is None:
                cells.append("—")
            else:
                # Both present — show head with base in parens
                # if the value moved. Same values collapse to a
                # single number so the table doesn't grow.
                if abs(head_val - base_val) < 1e-6:  # type: ignore[operator]
                    cells.append(_fmt_num(head_val))  # type: ignore[arg-type]
                else:
                    sign = "+" if head_val > base_val else ""  # type: ignore[operator]
                    delta = head_val - base_val  # type: ignore[operator]
                    cells.append(
                        f"{_fmt_num(head_val)} "  # type: ignore[arg-type]
                        f"({sign}{_fmt_num(delta)} from {_fmt_num(base_val)})"  # type: ignore[arg-type]
                    )
        body.append("| " + " | ".join(cells) + " |")

    parts: list[str] = []
    parts.append("### Eval leaderboard — selected-threshold diff")
    parts.append("")
    parts.append(
        "One row per config — the row the eval runner's best-threshold "
        "picker landed on. Numbers in **bold** are the new (head) value; "
        "the (delta from base) suffix shows the change since "
        "`origin/$BASE_BRANCH`."
    )
    parts.append("")
    if gate_thresholds is not None:
        parts.append(
            f"Gate: `hybrid_rrf` MRR ≥ {gate_thresholds.get('mrr_floor', 0):.2f} "
            f"& FPR-on-novel ≤ {gate_thresholds.get('fpr_ceiling', 0):.2f}."
        )
    parts.append(header_line)
    parts.append(sep_line)
    parts.extend(body)
    parts.append("")
    parts.append(
        "<sub>Generated by `.github/workflows/eval-regression.yml`. "
        "Numbers come from `results/leaderboard.csv`.</sub>"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="leaderboard_diff",
        description=(
            "Phase 3.6 — render a PR-friendly Markdown diff between the "
            "base branch's leaderboard.csv and the post-sweep head copy."
        ),
    )
    p.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Path to the base-branch leaderboard CSV (e.g. from `git show`).",
    )
    p.add_argument(
        "--head",
        type=Path,
        required=True,
        help="Path to the post-sweep leaderboard CSV.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Default: stdout.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.base.exists():
        print(f"[diff] base CSV not found: {args.base}", file=sys.stderr)
        return 1
    if not args.head.exists():
        print(f"[diff] head CSV not found: {args.head}", file=sys.stderr)
        return 1
    base_rows = read_leaderboard_csv(args.base)
    head_rows = read_leaderboard_csv(args.head)
    if not head_rows:
        print(f"[diff] head CSV is empty: {args.head}", file=sys.stderr)
        return 2
    md = render_diff(
        base_rows,
        head_rows,
        gate_thresholds={"mrr_floor": 0.40, "fpr_ceiling": 0.70},
    )
    if args.output is None:
        sys.stdout.write(md)
        sys.stdout.write("\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md + "\n", encoding="utf-8")
        print(f"[diff] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
