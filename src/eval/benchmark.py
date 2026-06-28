"""Benchmark loader for the eval harness.

A benchmark is a JSONL file where each line is one labeled record::

    {"id": "ev-001", "idea": "...", "expected_top_ids": [123, 456],
     "is_duplicate": true, "category": "duplicate", ...}

The full record schema is documented in ``docs/EVAL.md`` (the
construction policy lives there). Phase 1 uses the 100-idea
benchmark ``evals/labeled_v100.jsonl`` — see ``evals/labeled_v100.README.md``.

Why JSONL
---------
JSONL is line-oriented — one record per line, no trailing commas,
no top-level list bracket. This means:

- We can ``wc -l`` the benchmark to count records without parsing.
- A single bad line doesn't take down the whole file (the loader
  skips it with a clear warning — the runner reports the
  ``loaded / skipped / total`` counts so a partial load is loud).
- Easy to ``grep`` / ``jq 'select(.is_duplicate==false)'`` / pipe
  into pandas without wrapping the file.

``is_novel`` is the *inverse* of ``is_duplicate``. We expose both
because the runner's FPR-on-novel predicate uses ``is_novel`` (more
direct) and the leaderboard CSV's ``fpr_on_novel`` column header
matches the docs/EVAL.md terminology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BenchmarkRecord:
    """One row of the labeled benchmark.

    Fields
    ------
    id : str
        A stable record id, ``ev-NNN``. Used in error messages and
        in the per-record trace written to DuckDB.
    idea : str
        The free-text idea / query. What we POST to /search.
    expected_top_ids : tuple[int, ...]
        The company ids the label says are the "right answers"
        for this idea. Empty for novel records. Stored as a tuple
        so the record is hashable.
    is_duplicate : bool
        ``True`` if the idea has a known match in the corpus,
        ``False`` if it is genuinely novel.
    is_novel : bool
        Convenience inverse of ``is_duplicate``. Equivalent to
        ``not is_duplicate``. Used by the FPR-on-novel metric.
    category : str
        The label category. One of:
        ``duplicate``, ``novel``,
        ``adversarial_paraphrase``, ``adversarial_market_overlap``,
        ``adversarial_same_tech_diff_domain``, ``adversarial_temporal``.
        Free-form in Phase 1 — only used for the per-category
        breakdown (Phase 3) and for debugging.
    labeler : str
        Who labeled this record. ``anurag`` for the v100 set.
    labeled_at : str
        ISO timestamp of when the record was labeled. Free-form —
        not parsed.
    notes : str
        Free-form notes from the labeler. ``""`` if absent.
    """

    id: str
    idea: str
    expected_top_ids: tuple[int, ...]
    is_duplicate: bool
    is_novel: bool
    category: str
    labeler: str
    labeled_at: str
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "BenchmarkRecord":
        """Parse one JSONL row. Loud on missing required fields.

        Required keys: ``id``, ``idea``, ``expected_top_ids``,
        ``is_duplicate``, ``category``. Optional: ``labeler``,
        ``labeled_at``, ``notes``.
        """
        for key in ("id", "idea", "expected_top_ids", "is_duplicate", "category"):
            if key not in raw:
                raise ValueError(f"benchmark record missing key '{key}': {raw!r}")

        expected = raw["expected_top_ids"]
        if not isinstance(expected, list):
            raise ValueError(
                f"record {raw.get('id', '?')}: expected_top_ids must be a list, got {type(expected)}"
            )

        return cls(
            id=str(raw["id"]),
            idea=str(raw["idea"]),
            expected_top_ids=tuple(int(x) for x in expected),
            is_duplicate=bool(raw["is_duplicate"]),
            is_novel=not bool(raw["is_duplicate"]),
            category=str(raw["category"]),
            labeler=str(raw.get("labeler", "unknown")),
            labeled_at=str(raw.get("labeled_at", "")),
            notes=str(raw.get("notes", "")),
        )


@dataclass
class Benchmark:
    """A loaded benchmark — a list of records + the path it came from."""

    records: List[BenchmarkRecord]
    path: Path

    def __len__(self) -> int:
        return len(self.records)

    def novel_records(self) -> List[BenchmarkRecord]:
        """Records labeled ``is_duplicate=False``."""
        return [r for r in self.records if r.is_novel]

    def duplicate_records(self) -> List[BenchmarkRecord]:
        """Records labeled ``is_duplicate=True``."""
        return [r for r in self.records if r.is_duplicate]


class BenchmarkLoadError(Exception):
    """Raised when the benchmark file can't be loaded (missing,
    not JSONL, etc.). Distinct from per-record parse errors which
    are logged and skipped (see ``load_benchmark``)."""


def load_benchmark(
    path: Path,
    *,
    skip_invalid: bool = True,
) -> Benchmark:
    """Load a JSONL benchmark from disk.

    Parameters
    ----------
    path : Path
        The JSONL file. One record per line.
    skip_invalid : bool, default True
        If True, per-record parse errors are logged via ``print``
        and the record is skipped (the runner reports
        ``loaded=N skipped=M``). If False, the first error raises
        ``ValueError`` and aborts the load. Use False in tests
        where a single bad line is a test failure.

    Returns
    -------
    Benchmark
        A loaded benchmark ready to iterate over.

    Raises
    ------
    BenchmarkLoadError
        If the file is missing, not a file, or completely empty.
    """
    if not path.exists():
        raise BenchmarkLoadError(f"benchmark file not found: {path}")
    if not path.is_file():
        raise BenchmarkLoadError(f"benchmark path is not a file: {path}")

    records: List[BenchmarkRecord] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # blank line — ignore
            try:
                raw = json.loads(line)
                rec = BenchmarkRecord.from_dict(raw)
                records.append(rec)
            except (ValueError, json.JSONDecodeError) as exc:
                skipped += 1
                if not skip_invalid:
                    raise
                print(f"[benchmark] WARN {path}:{lineno}: skipped: {exc}")

    if not records:
        raise BenchmarkLoadError(f"benchmark file is empty (or all lines invalid): {path}")

    if skipped:
        print(f"[benchmark] loaded {len(records)} records from {path} ({skipped} skipped)")
    return Benchmark(records=records, path=path)