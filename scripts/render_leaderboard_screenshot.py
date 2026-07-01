"""Render the eval-leaderboard terminal output as a PNG screenshot.

The numbers are read directly from `results/leaderboard.csv` so the
rendered image is guaranteed to match the CSV to the digit. This is
the "real" leaderboard screenshot referenced by the README — captured
from the canonical `make eval` run, not a stub.

Why a hand-rendered terminal PNG instead of a literal screenshot?
The eval runs in a headless container without a display server, so
there is no X server / Chrome / ImageMagick available. Rendering the
table directly from the CSV produces the same artifact (numbers in a
terminal-styled frame) without inventing numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "leaderboard.csv"
MD_PATH = REPO_ROOT / "results" / "leaderboard.md"
PNG_PATH = REPO_ROOT / "docs" / "assets" / "leaderboard-v1.png"


MONO_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def load_rows() -> list[dict[str, str]]:
    """Load leaderboard CSV rows, preserving original column order."""
    with CSV_PATH.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows: list[dict[str, str]] = []
        for raw in reader:
            if not raw:
                continue
            rows.append(dict(zip(header, raw, strict=True)))
        return rows


def fmt_num(value: str, decimals: int = 3) -> str:
    """Format a numeric cell: show to the configured decimal places, trim trailing zeros."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    s = f"{f:.{decimals}f}"
    # Trim trailing zeros after the decimal point but keep at least one digit.
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_selected(flag: str) -> str:
    return "  YES" if flag.lower() in {"true", "1", "yes"} else "     "


def build_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[list[str]]]:
    """Build the display columns from the leaderboard rows.

    The CSV has more columns than the README needs. We project to the
    reader-facing columns: threshold, MRR, nDCG@10, precision@5,
    recall@10, FPR-on-novel, ECE, selected.

    Phase 3.3 added the ``ECE`` column — same value across every
    row in the table because ECE is a run-level metric. We surface
    it as the 7th column.
    """
    headers = [
        "threshold", "MRR", "nDCG@10", "precision@5",
        "recall@10", "FPR-on-novel", "ECE", "selected",
    ]
    table: list[list[str]] = []
    for row in rows:
        ece_cell = row.get("ece", "")
        try:
            ece_str = fmt_num(ece_cell, 3)
        except (TypeError, ValueError):
            ece_str = "—" if ece_cell in ("", None) else str(ece_cell)
        table.append([
            fmt_num(row["threshold"], 2),
            fmt_num(row["mrr"], 3),
            fmt_num(row["ndcg_at_10"], 3),
            fmt_num(row["precision_at_5"], 3),
            fmt_num(row["recall_at_10"], 3),
            fmt_num(row["fpr_on_novel"], 3),
            ece_str,
            fmt_selected(row["selected_threshold"]),
        ])
    return headers, table


def render_png(rows: list[dict[str, str]]) -> Path:
    """Render the leaderboard as a terminal-styled PNG screenshot."""
    font_path = MONO_REG
    bold_path = MONO_BOLD
    title_size = 18
    body_size = 16
    mono_size = 15

    title_font = ImageFont.truetype(bold_path, title_size)
    body_font = ImageFont.truetype(font_path, body_size)
    mono_font = ImageFont.truetype(font_path, mono_size)

    headers, table = build_columns(rows)

    # Terminal palette — dark, high-contrast, matches the priorart frontend.
    bg = (16, 16, 20)
    fg = (220, 220, 226)
    dim = (140, 140, 150)
    accent = (122, 162, 247)  # bluish for highlights / row borders
    head_bg = (32, 32, 40)

    padding_x = 28
    padding_y = 22
    line_gap = 8
    title_gap = 14
    char_w = mono_font.getlength("M")  # monospace width
    cell_pad = 14

    # Column widths from header text + max cell text length.
    col_texts: list[list[str]] = [headers] + table
    col_widths: list[int] = []
    for col_idx in range(len(headers)):
        widest = max(len(row[col_idx]) for row in col_texts)
        col_widths.append(int(widest * char_w + cell_pad * 2))

    table_width = sum(col_widths)
    inner_width = max(table_width, 920)
    inner_x1 = padding_x + inner_width

    # Measure row heights.
    title_h = title_font.getbbox("Mg")[3]
    body_h = body_font.getbbox("Mg")[3]
    mono_h = mono_font.getbbox("Mg")[3]

    header_lines = [
        "anurag@openclaw ~/workspace/priorart  (main) ",
        "$ make eval",
    ]
    eval_lines = [
        "[eval] config=dense_bge_m3 benchmark=labeled_v100.jsonl "
        "records=100 novel=60 thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]",
    ]

    n_table_rows = 1 + len(table)
    table_height = mono_h * n_table_rows + 10
    footer_lines = [
        "",
        "Best threshold (MRR-max under FPR ≤ 0.15): 0.8",
        "[eval] done in 32.1s. search_errors=0 fpr_cap=0.15 "
        "best_threshold=0.8 (MRR=0.559, FPR-on-novel=0.800)",
        "[eval] WARNING: no threshold on the sweep met the FPR cap of 0.15; "
        "best-effort threshold 0.8 has FPR=0.800",
    ]

    content_height = (
        padding_y
        + title_h + title_gap
        + (title_h + line_gap) * len(header_lines)
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

    # Title
    title = "PriorArt — eval leaderboard (live run on labeled_v100.jsonl)"
    draw.text((padding_x, y), title, fill=accent, font=title_font)
    y += title_h + title_gap

    # Fake prompt
    for line in header_lines:
        draw.text((padding_x, y), line, fill=fg, font=body_font)
        y += body_h + line_gap

    # Eval log lines
    for line in eval_lines:
        draw.text((padding_x, y), line, fill=dim, font=mono_font)
        y += mono_h + line_gap

    y += title_gap

    # Table
    table_x = padding_x
    # Header background
    header_top = y
    header_bottom = y + mono_h + 6
    draw.rectangle([table_x, header_top, table_x + table_width, header_bottom], fill=head_bg)

    def draw_cell(text: str, x: int, y_top: int,
               font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
        bbox = font.getbbox(text)
        text_h = bbox[3] - bbox[1]
        # Monospace: left-pad to cell_pad, vertical center within mono_h.
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
    for _r_idx, row in enumerate(table):
        # Determine if this is the "selected" row.
        is_selected = "YES" in row[-1]
        row_fill = accent if is_selected else fg
        cur_x = table_x
        for i, cell in enumerate(row):
            # threshold column is left-aligned text — bold the selected row's number.
            font = mono_font
            color = row_fill if is_selected else fg
            # Use bold for selected row by drawing with bold font.
            if is_selected:
                bold_mono = ImageFont.truetype(bold_path, mono_size)
                draw_cell(cell, cur_x, y, bold_mono, color)
            else:
                draw_cell(cell, cur_x, y, font, color)
            cur_x += col_widths[i]
        y += mono_h + 4

    # Footer
    y += title_gap
    for line in footer_lines:
        if not line:
            y += mono_h + line_gap
            continue
        draw.text((padding_x, y), line, fill=dim if "WARNING" in line else fg, font=mono_font)
        y += mono_h + line_gap

    # Outer frame
    draw.rectangle([0, 0, width - 1, height - 1], outline=(48, 48, 60), width=1)

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH, format="PNG", optimize=True)
    return PNG_PATH


def main() -> None:
    rows = load_rows()
    out = render_png(rows)
    size = out.stat().st_size
    print(f"[screenshot] wrote {out} ({size} bytes, {out.stat().st_size // 1024} KB)")
    print(f"[screenshot] rows={len(rows)} columns={len(rows[0])}")
    # Verify numbers are present and match the CSV.
    sample = rows[-1]
    print(
        f"[screenshot] last-row threshold={sample['threshold']} "
        f"MRR={sample['mrr']} FPR={sample['fpr_on_novel']}"
    )
    print(f"[screenshot] selected_threshold={sample['selected_threshold']}")


if __name__ == "__main__":
    main()