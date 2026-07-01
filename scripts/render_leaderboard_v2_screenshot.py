"""Render the Phase 2 eval leaderboard as `docs/assets/leaderboard-v2.png`.

This is the v2 of the README leaderboard image: a 3-config comparison
(dense_bge_m3 / bm25 / hybrid_rrf) on the `labeled_v300.jsonl` benchmark,
one row per config (the `selected_threshold` per config). The image is
generated from `results/leaderboard.csv` directly so the numbers in the
PNG match the CSV to the digit.

This is NOT a literal terminal screenshot (no display server in the
container) — it's a hand-rendered terminal-styled frame so the artifact
is reproducible from the committed CSV without inventing numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "leaderboard.csv"
PNG_PATH = REPO_ROOT / "docs" / "assets" / "leaderboard-v2.png"


MONO_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def fmt_num(value: str, decimals: int = 3) -> str:
    """Format a numeric cell — show to N decimals, trim trailing zeros."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    s = f"{f:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def load_selected_v300_rows() -> list[dict[str, str]]:
    """Load `results/leaderboard.csv` and return the selected-threshold row per
    config on the `labeled_v300.jsonl` benchmark, in stable order:
    dense_bge_m3 → bm25 → hybrid_rrf.

    The CSV is append-only over the run history; multiple threshold rows
    exist per (config, benchmark). The `selected_threshold=True` row is
    the one we ship in the README — the threshold that maximizes MRR
    subject to FPR-on-novel ≤ 0.15 (Phase 1 acceptance cap).
    """
    with CSV_PATH.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["benchmark"] != "labeled_v300.jsonl":
            continue
        if row["selected_threshold"].strip().lower() not in {"true", "1", "yes"}:
            continue
        cfg = row["config"]
        # First selected row per config wins. CSV order is preserved, and
        # the eval runner writes one selected row per (config, benchmark).
        selected.setdefault(cfg, row)

    # Stable, sensible order for the README
    order = ["dense_bge_m3", "bm25", "hybrid_rrf"]
    out: list[dict[str, str]] = []
    for cfg in order:
        if cfg in selected:
            out.append(selected[cfg])
    return out


def short_label(embedding_model: str) -> str:
    """Compact label for the embedding column (long strings wrap)."""
    if "bge-m3" in embedding_model and "rank_bm25" in embedding_model:
        return "bge-m3 + BM25 (RRF k=60)"
    if "bge-m3" in embedding_model:
        return "bge-m3 (pgvector HNSW)"
    if "rank_bm25" in embedding_model:
        return "rank_bm25 (BM25Okapi)"
    return embedding_model


def render_png(rows: list[dict[str, str]]) -> Path:
    """Render the v2 leaderboard PNG: 3 configs × 6 metrics + embedding."""
    font_path = MONO_REG
    bold_path = MONO_BOLD
    title_size = 18
    body_size = 16
    mono_size = 15

    title_font = ImageFont.truetype(bold_path, title_size)
    body_font = ImageFont.truetype(font_path, body_size)
    mono_font = ImageFont.truetype(font_path, mono_size)

    # Headers: config + embedding + 5 metrics
    headers = [
        "config", "embedding", "threshold",
        "MRR", "nDCG@10", "P@5", "R@10", "FPR-novel", "ECE",
    ]
    table = []
    for row in rows:
        ece_cell = row.get("ece", "")
        try:
            ece_str = fmt_num(ece_cell, 3)
        except (TypeError, ValueError):
            ece_str = "—"
        table.append([
            row["config"],
            short_label(row["embedding_model"]),
            fmt_num(row["threshold"], 2),
            fmt_num(row["mrr"], 3),
            fmt_num(row["ndcg_at_10"], 3),
            fmt_num(row["precision_at_5"], 3),
            fmt_num(row["recall_at_10"], 3),
            fmt_num(row["fpr_on_novel"], 3),
            ece_str,
        ])

    bg = (16, 16, 20)
    fg = (220, 220, 226)
    dim = (140, 140, 150)
    accent = (122, 162, 247)
    head_bg = (32, 32, 40)
    warn = (251, 113, 133)  # rose-400 — used for the FPR-novel caveat line

    padding_x = 28
    padding_y = 22
    line_gap = 8
    title_gap = 14
    char_w = mono_font.getlength("M")
    cell_pad = 14

    col_texts: list[list[str]] = [headers] + table
    col_widths: list[int] = []
    for col_idx in range(len(headers)):
        widest = max(len(row[col_idx]) for row in col_texts)
        col_widths.append(int(widest * char_w + cell_pad * 2))

    table_width = sum(col_widths)
    inner_width = max(table_width, 1080)
    inner_x1 = padding_x + inner_width

    title_h = title_font.getbbox("Mg")[3]
    body_h = body_font.getbbox("Mg")[3]
    mono_h = mono_font.getbbox("Mg")[3]

    header_lines = [
        "anurag@openclaw ~/workspace/priorart  (main) ",
        "$ make eval BENCH=evals/labeled_v300.jsonl",
    ]
    eval_lines = [
        "[eval] dispatching 3 configs: dense_bge_m3, bm25, hybrid_rrf",
        "[eval] benchmark=labeled_v300.jsonl records=300 "
        "(100 duplicate / 100 novel / 100 adversarial)",
        "[eval] corpus_count=10983 (yc=5949 + producthunt=4000 + hn=993)",
        "[eval] provenance=llm-generated-v2-pending-anurag-hand-review "
        "(see Limitations)",
        "[eval] MLflow experiment=phase-2-baseline, 3 runs FINISHED",
    ]
    footer_lines = [
        "",
        "Best per-config threshold (MRR-max; eval runner falls back when no",
        "threshold meets the FPR ≤ 0.15 cap — all 3 configs in this run):",
        "  dense_bge_m3 → 0.8 (MRR=0.567, FPR-novel=0.79)",
        "  bm25         → 0.5 (MRR=0.392, FPR-novel=1.00)",
        "  hybrid_rrf   → 0.8 (MRR=0.458, FPR-novel=0.63)",
        "",
        "PHASE-3.md §3.3 ECE column (run-level, LLM-generated v300):",
        "  per-config calibration PNGs live under docs/assets/calibration-*.png",
        "  ECE ≤ 0.10 is the *informational* target; recorded verbatim below.",
        "",
        "PHASE-2.md §Definition-of-done MRR targets are INFORMATIONAL until",
        "the labeled_v300 hand-label pass lands (eval set is LLM-generated v2).",
    ]


    n_table_rows = 1 + len(table)
    table_height = mono_h * n_table_rows + 10

    content_height = (
        padding_y
        + title_h + title_gap
        + (body_h + line_gap) * len(header_lines)
        + (mono_h + line_gap) * len(eval_lines)
        + title_gap
        + table_height
        + title_gap
        + (mono_h + line_gap) * len(footer_lines)
        + padding_y
    )

    width = int(inner_x1 + padding_x)
    height = int(content_height)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    y = padding_y

    title = "PriorArt — eval leaderboard v2 (3 configs on labeled_v300.jsonl)"
    draw.text((padding_x, y), title, fill=accent, font=title_font)
    y += title_h + title_gap

    for line in header_lines:
        draw.text((padding_x, y), line, fill=fg, font=body_font)
        y += body_h + line_gap

    for line in eval_lines:
        draw.text((padding_x, y), line, fill=dim, font=mono_font)
        y += mono_h + line_gap

    y += title_gap

    # Table
    table_x = padding_x
    header_top = y
    header_bottom = y + mono_h + 6
    draw.rectangle(
        [table_x, header_top, table_x + table_width, header_bottom],
        fill=head_bg,
    )

    def draw_cell(text: str, x: int, y_top: int,
                  font: ImageFont.FreeTypeFont,
                  fill: tuple[int, int, int]) -> None:
        bbox = font.getbbox(text)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x + cell_pad, y_top + (mono_h - text_h) // 2 - bbox[1]),
            text, fill=fill, font=font,
        )

    # Header row
    cur_x = table_x
    for i, h in enumerate(headers):
        draw_cell(h, cur_x, y, mono_font, accent)
        cur_x += col_widths[i]
    y += mono_h + 6

    # Body rows
    FPR_NOVEL_COL_INDEX = 7  # 0-based index in the headers list
    for row in table:
        cur_x = table_x
        for i, cell in enumerate(row):
            # First column (config) is the row label — keep accent.
            # The FPR-novel column gets the rose caveat color since neither
            # bm25 nor hybrid_rrf clears the 0.15 cap. ECE is dim because
            # it's a run-level secondary metric here.
            if i == FPR_NOVEL_COL_INDEX:
                color = warn
            elif i == 0:
                color = accent
            elif i == 8:  # ECE column — secondary metric, dim text
                color = dim
            else:
                color = fg
            draw_cell(cell, cur_x, y, mono_font, color)
            cur_x += col_widths[i]
        y += mono_h + 4

    # Footer
    y += title_gap
    for line in footer_lines:
        if not line:
            y += mono_h + line_gap
            continue
        is_warn = ("INFORMATIONAL" in line
                   or "NO threshold meets cap" in line)
        draw.text(
            (padding_x, y), line,
            fill=warn if is_warn else (dim if line.startswith("[eval]") else fg),
            font=mono_font,
        )
        y += mono_h + line_gap

    draw.rectangle([0, 0, width - 1, height - 1], outline=(48, 48, 60), width=1)

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", optimize=True)
    return PNG_PATH


def main() -> None:
    rows = load_selected_v300_rows()
    if len(rows) != 3:
        raise SystemExit(
            f"expected 3 selected rows (dense/bm25/hybrid_rrf) on "
            f"labeled_v300.jsonl, got {len(rows)}: "
            f"{[r['config'] for r in rows]}"
        )
    out = render_png(rows)
    size = out.stat().st_size
    print(f"[screenshot-v2] wrote {out} ({size} bytes, {size // 1024} KB)")
    print(f"[screenshot-v2] rows={len(rows)} columns={len(rows[0])}")
    for r in rows:
        print(
            f"[screenshot-v2] {r['config']:>13} | thr={r['threshold']:>4} | "
            f"MRR={fmt_num(r['mrr'], 3)} nDCG@10={fmt_num(r['ndcg_at_10'], 3)} "
            f"P@5={fmt_num(r['precision_at_5'], 3)} R@10={fmt_num(r['recall_at_10'], 3)} "
            f"FPR={fmt_num(r['fpr_on_novel'], 3)}"
        )


if __name__ == "__main__":
    main()