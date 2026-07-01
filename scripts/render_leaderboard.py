"""Render `results/leaderboard.md` from `results/leaderboard.csv`.

The eval runner (``python -m eval.run --markdown-out ...``) only writes a
per-run Markdown summary. The canonical leaderboard Markdown — the one
the README links to and that humans eyeball first — is regenerated from
the CSV by this script so it stays in sync with whatever the runner
appends.

Idempotent. Run after every eval-run append. Safe to run multiple times.

Usage::

    uv run python scripts/render_leaderboard.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "leaderboard.csv"
OUT = ROOT / "results" / "leaderboard.md"

# Config ordering for the section list — dense first (baseline), then BM25
# (lexical floor), then hybrid (ensemble). Unknown configs sort last.
_CFG_ORDER = {"dense_bge_m3": 0, "bm25": 1, "hybrid_rrf": 2}


def render() -> str:
    with CSV.open() as f:
        rows = list(csv.DictReader(f))

    # Group by (config, benchmark, corpus_count, embedding_model)
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["config"], r["benchmark"], r["corpus_count"], r["embedding_model"])
        groups[key].append(r)

    def sort_key(item):
        cfg, bench, _, _ = item[0]
        return (_CFG_ORDER.get(cfg, 99), bench)

    lines: list[str] = []
    lines.append("# PriorArt Eval Leaderboard")
    lines.append("")
    lines.append(
        "Generated from `results/leaderboard.csv`. Rows are grouped by "
        "(config, benchmark) and sorted by threshold. The `selected` row is "
        "the threshold that maximises MRR subject to FPR-on-novel ≤ 0.15 "
        "(Phase 1 acceptance cap)."
    )
    lines.append("")
    lines.append(
        "Eval set provenance: `evals/labeled_v300.jsonl` is **LLM-generated v2 "
        "with honest provenance** (`labeler=ai-assisted-claude-minimax-m3`, "
        "`provenance=llm-generated-v2-pending-anurag-hand-review`) per the "
        "Phase 1.5a fix (commit c8aa1fb). MRR targets in PHASE-2.md "
        "§Definition-of-done are INFORMATIONAL until the hand-label pass lands."
    )
    lines.append("")
    # ECE provenance call-out. Mirrors the PNG title so anyone
    # reading either artifact sees the same honest statement.
    lines.append(
        "<!-- ECE computed against LLM-generated v300; hand-label pending -->\n"
        "<!-- ECE ≤ 0.10 is the PHASE-3.md §3.3 *informational* target; recorded verbatim below. -->"
    )
    lines.append("")
    lines.append(
        "Cohere rerank is opt-in only (AGENTS.md + PHASE-2.md §Pitfalls) and "
        "NOT a default Phase 2 config — not present in this table."
    )
    lines.append("")

    for (cfg, bench, corpus, model), g in sorted(groups.items(), key=sort_key):
        lines.append(f"## `{cfg}` on `{bench}` — corpus={corpus} ({model})")
        lines.append("")
        lines.append(
            "| threshold | MRR | nDCG@10 | precision@5 | recall@10 | "
            "FPR-on-novel | ECE | selected |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(g, key=lambda x: float(x["threshold"])):
            sel = "**YES**" if r["selected_threshold"] == "True" else ""
            ece_cell = r.get("ece", "")
            try:
                ece_str = f"{float(ece_cell):.3f}"
            except (TypeError, ValueError):
                ece_str = "—" if ece_cell in ("", None) else str(ece_cell)
            lines.append(
                f"| {r['threshold']} | {r['mrr']} | {r['ndcg_at_10']} | "
                f"{r['precision_at_5']} | {r['recall_at_10']} | "
                f"{r['fpr_on_novel']} | {ece_str} | {sel} |"
            )
        bests = [r for r in g if r["selected_threshold"] == "True"]
        if bests:
            b = bests[0]
            ece_cell = b.get("ece", "")
            try:
                ece_str = f"{float(ece_cell):.3f}"
            except (TypeError, ValueError):
                ece_str = "—" if ece_cell in ("", None) else str(ece_cell)
            lines.append("")
            lines.append(
                f"Best threshold (MRR-max under FPR ≤ 0.15): **{b['threshold']}** "
                f"— MRR={b['mrr']}, FPR-on-novel={b['fpr_on_novel']}, "
                f"ECE={ece_str}"
            )
        lines.append("")
        note = (g[0].get("notes") or "").strip().replace("\n", " ")
        if note:
            lines.append(f"_Notes: {note[:300]}_")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    md = render()
    OUT.write_text(md)
    print(f"wrote {len(md)} bytes to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())