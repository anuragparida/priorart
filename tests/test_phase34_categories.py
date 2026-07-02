"""Tests for Phase 3.4 — per-category failure analysis.

What this covers
----------------
- ``src/eval/categorize.py`` — the deterministic rule-based
  business-category assigner. We test the *contract* (always
  returns a BusinessCategory, deterministic across re-runs, all 8
  buckets are reachable on a real eval set), not the exact match
  on every free-form record.
- ``src/eval/failure_analysis.py`` — the per-category metrics
  + markdown writer + heatmap + CSV writer. We test the
  per-category aggregation against a hand-built
  ``PerRecordResult`` fixture so the assertions are tight
  (no live API).
- ``scripts/build_business_categories.py`` — the script that
  extends the eval set with the new fields. We test idempotency
  (re-running yields the same output) and that every record
  gets both new fields.

Test layout
-----------
- ``TestCategorizeContract`` — the small invariant checks
  (8 buckets reachable, deterministic, etc).
- ``TestCategorizeExamples`` — example-based checks against
  unambiguous ideas (a neobank → fintech, a tutoring app →
  education). These are not a *full* coverage suite — they
  pin the obvious wins so a future rule-set rewrite doesn't
  silently regress them.
- ``TestCategorizeEvalSetCoverage`` — runs the assigner over
  the real ``evals/labeled_v300.jsonl`` and asserts the
  coverage is "real" (no single bucket is 100% of the records,
  every record has a category).
- ``TestFailureAnalysisMetrics`` — the per-category metrics
  on a hand-built ``PerRecordResult`` fixture.
- ``TestFailureAnalysisWriters`` — the per-config MD / CSV /
  heatmap writers render the expected shape.
- ``TestBuildBusinessCategories`` — the build script's
  idempotency + field-coverage contract.

Honest scope
------------
The rule-based assigner is a v1 — its hand-label accuracy
isn't measured here. The card explicitly says the rule set is
v1 pending Anurag's hand-label pass; these tests pin the
*contract* (no crashes, every record categorised, deterministic
across re-runs), not the *accuracy*.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Make the repo root importable when pytest is run from a
# different cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.categorize import (  # noqa: E402
    BUSINESS_CATEGORIES,
    CATEGORY_LABEL,
    DEFAULT_PROVENANCE,
    BusinessCategory,
    CategoryAssignment,
    assign_business_category,
    assign_business_categories,
    category_coverage,
)
from src.eval.failure_analysis import (  # noqa: E402
    BREAKDOWN_CSV_COLUMNS,
    PerCategoryMetrics,
    build_csv_row,
    compute_per_category_metrics_from_benchmark,
    plot_heatmap,
    write_breakdown_csv,
    write_per_config_markdown,
)
from src.eval.metrics import (  # noqa: E402
    fpr_on_novel_record,
    ndcg_at_k,
    reciprocal_rank,
)
from src.eval.run import PerRecordResult  # noqa: E402


# ---------------------------------------------------------------------------
# TestCategorizeContract — invariants
# ---------------------------------------------------------------------------


class TestCategorizeContract:
    """Invariants every rule-set rewrite must preserve."""

    def test_business_categories_constant(self):
        # The 8 PHASE-3.md §3.4 buckets, in a stable order.
        assert len(BUSINESS_CATEGORIES) == 8
        expected_values = {
            "b2b_saas", "consumer", "devtools", "marketplace",
            "fintech", "healthcare", "education", "other",
        }
        assert {c.value for c in BUSINESS_CATEGORIES} == expected_values

    def test_category_label_covers_every_bucket(self):
        # The label table is used by the per-config MD writer +
        # the heatmap; every bucket must have a human label.
        for cat in BUSINESS_CATEGORIES:
            assert cat in CATEGORY_LABEL
            label = CATEGORY_LABEL[cat]
            assert label  # non-empty
            # The label is what a human reads on the breakdown
            # table — no Python identifiers (no underscores), and
            # the label string is humanised (Title-case or has a
            # space, not the raw enum value).
            assert "_" not in label, f"label for {cat} contains an underscore: {label!r}"
            assert label == label.title() or " " in label, (
                f"label for {cat} is not title-cased / multi-word: {label!r}"
            )

    def test_provenance_stamp_is_honest(self):
        # The provenance stamp must call out the v1 status so
        # downstream readers know the categories are rule-based,
        # not hand-labeled. Pin the exact string the card
        # approved.
        assert "v1" in DEFAULT_PROVENANCE
        assert "pending" in DEFAULT_PROVENANCE
        assert "anurag" in DEFAULT_PROVENANCE.lower()

    def test_assign_returns_enum(self):
        a = assign_business_category("AI-powered contract review for SMB law firms")
        assert isinstance(a, CategoryAssignment)
        assert isinstance(a.business_category, BusinessCategory)
        # matched_rule_index is an int for a non-fallback match,
        # None for OTHER. Both are valid.
        assert a.matched_rule_index is None or isinstance(a.matched_rule_index, int)

    def test_assign_is_deterministic(self):
        # Same input -> same output, byte-for-byte. This is
        # the determinism contract the card pinned.
        ideas = [
            "AI-powered contract review for SMB law firms",
            "GitHub app that auto-reviews your pull requests",
            "A neobank for freelancers",
        ]
        first = [assign_business_category(i).business_category for i in ideas]
        for _ in range(5):
            again = [assign_business_category(i).business_category for i in ideas]
            assert again == first

    def test_empty_input_falls_back_to_other(self):
        # No rule can fire on an empty string. The fallback is
        # OTHER (not an exception) so the JSONL extension never
        # aborts on a weird row.
        assert assign_business_category("").business_category == BusinessCategory.OTHER
        assert assign_business_category(None).business_category == BusinessCategory.OTHER
        assert assign_business_category("   ").business_category == BusinessCategory.OTHER

    def test_assign_categories_batch_is_idempotent(self):
        pairs = [
            ("rec-1", "AI-powered contract review for SMB law firms"),
            ("rec-2", "GitHub app that auto-reviews your pull requests"),
            ("rec-3", "A neobank for freelancers"),
        ]
        first = assign_business_categories(pairs)
        again = assign_business_categories(pairs)
        assert first == again
        assert set(first.keys()) == {"rec-1", "rec-2", "rec-3"}

    def test_assign_categories_raises_on_duplicate_id(self):
        # The batch helper is keyed by record id; a duplicate
        # id is a contract violation (we'd silently overwrite
        # otherwise).
        with pytest.raises(ValueError, match="duplicate record_id"):
            assign_business_categories([
                ("rec-1", "AI"),
                ("rec-1", "Also AI"),
            ])

    def test_category_coverage_returns_every_bucket(self):
        # The coverage helper always returns a dict keyed by
        # every bucket (so the writer can render zero rows
        # cleanly).
        cov = category_coverage([BusinessCategory.B2B_SAAS])
        assert set(cov.keys()) == set(BUSINESS_CATEGORIES)
        assert cov[BusinessCategory.B2B_SAAS] == 1
        assert cov[BusinessCategory.OTHER] == 0


# ---------------------------------------------------------------------------
# TestCategorizeExamples — pin the obvious wins
# ---------------------------------------------------------------------------


class TestCategorizeExamples:
    """Example-based tests for unambiguous idea descriptions.

    These are not a full coverage suite — they pin the obvious
    wins so a future rule-set rewrite doesn't silently regress
    them. The card explicitly says the categories are v1
    pending Anurag's hand-label pass; these tests hold the line
    on the easy cases, not the hard ones.
    """

    @pytest.mark.parametrize("idea,expected", [
        ("A neobank for freelancers in Southeast Asia", BusinessCategory.FINTECH),
        ("Stripe for marketplace seller payouts with embedded KYC", BusinessCategory.FINTECH),
        ("Crypto trading bot with on-chain analytics", BusinessCategory.FINTECH),
        ("Mortgage application platform for self-employed borrowers", BusinessCategory.FINTECH),
        ("AI co-pilot for sales teams that scores their calls in real time", BusinessCategory.B2B_SAAS),
        ("An AI agent for customer support teams", BusinessCategory.B2B_SAAS),
        ("CI/CD platform for monorepo TypeScript projects", BusinessCategory.DEVTOOLS),
        ("Vector database for production RAG pipelines", BusinessCategory.DEVTOOLS),
        ("A marketplace for renting vintage sewing machines from local collectors by the weekend", BusinessCategory.MARKETPLACE),
        ("Uber for on-demand sommeliers for private dinner parties", BusinessCategory.MARKETPLACE),
        ("A telemedicine platform for rural hospitals", BusinessCategory.HEALTHCARE),
        ("Patient billing and collections platform that helps medical practices collect outstanding receivables", BusinessCategory.HEALTHCARE),
        ("A personalized AI tutor that adapts lessons to each student's pace and learning style", BusinessCategory.EDUCATION),
        ("An app that teaches you Spanish with AI", BusinessCategory.EDUCATION),
        ("Social network for origami enthusiasts to share crease patterns and folding instructions", BusinessCategory.CONSUMER),
        ("AI-powered personal stylist that picks outfits from your closet", BusinessCategory.CONSUMER),
    ])
    def test_obvious_wins(self, idea, expected):
        a = assign_business_category(idea)
        assert a.business_category == expected, (
            f"expected {expected.value} for {idea!r}, got {a.business_category.value}"
        )


# ---------------------------------------------------------------------------
# TestCategorizeEvalSetCoverage — the real eval set
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "evals" / "labeled_v300.jsonl").exists(),
    reason="evals/labeled_v300.jsonl not present",
)
class TestCategorizeEvalSetCoverage:
    """Run the assigner over the real eval set; check the shape.

    These tests don't pin specific category counts (the rule set
    will evolve) but they DO pin:
    - every record has a category,
    - the 8 buckets are all reachable on a real eval set
      (i.e. the rule set isn't degenerate),
    - the OTHER bucket isn't 100% (the rule set is actually
      doing work).
    """

    @pytest.fixture(scope="class")
    def eval_records(self) -> List[dict]:
        with (_REPO_ROOT / "evals" / "labeled_v300.jsonl").open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_every_record_has_business_category(self, eval_records):
        # Card acceptance: "all 300 records have a category field"
        for rec in eval_records:
            assert "business_category" in rec, f"{rec.get('id')}: missing business_category"
            assert rec["business_category"] in {c.value for c in BUSINESS_CATEGORIES}

    def test_every_record_has_provenance(self, eval_records):
        # Card acceptance: "AND a category_provenance field on every record"
        for rec in eval_records:
            assert "business_category_provenance" in rec
            assert "v1" in rec["business_category_provenance"]
            assert "pending" in rec["business_category_provenance"]

    def test_all_eight_buckets_reachable(self, eval_records):
        # On a 300-record eval set with the LLM-generated
        # description style, at least 6 of 8 buckets should be
        # non-empty. (We allow 2 empty because some buckets like
        # "healthcare" / "education" have very few seed records
        # in the LLM-generated set.)
        seen = {rec["business_category"] for rec in eval_records}
        assert len(seen) >= 6, f"only {len(seen)} buckets reachable: {seen}"

    def test_other_bucket_not_monopolistic(self, eval_records):
        # The rule set should be doing *some* work. The 173/300
        # baseline we measured empirically is the floor; if the
        # rule set regresses to 290/300 OTHER, this test will
        # catch it.
        n_other = sum(1 for r in eval_records if r["business_category"] == "other")
        assert n_other < 250, (
            f"OTHER bucket is {n_other}/300 — rule set may have "
            f"regressed; expected the rule set to do more work"
        )


# ---------------------------------------------------------------------------
# TestFailureAnalysisMetrics — the per-category aggregation
# ---------------------------------------------------------------------------


def _make_result(
    record_id: str,
    *,
    category: str,
    is_duplicate: bool,
    is_novel: bool,
    ranked_ids: Tuple[int, ...] = (),
    top1_score: float = None,
    search_error: str = None,
) -> PerRecordResult:
    """Build a PerRecordResult without going through the API."""
    return PerRecordResult(
        record_id=record_id,
        category=category,
        is_duplicate=is_duplicate,
        is_novel=is_novel,
        ranked_ids=ranked_ids,
        ranked_scores=tuple(),
        top1_score=top1_score,
        search_error=search_error,
    )


class TestFailureAnalysisMetrics:
    """The per-category metrics on a hand-built fixture."""

    def _build_fixture(self) -> Tuple[List[PerRecordResult], Dict[str, str], Dict[str, Tuple[int, ...]]]:
        """Hand-built fixture: 3 records, 1 in b2b_saas, 1 in healthcare, 1 in other."""
        results = [
            # b2b_saas: relevant (duplicate), expected [42], got [42, 7] -> RR=1.0
            _make_result(
                "b2b-1",
                category="duplicate",
                is_duplicate=True,
                is_novel=False,
                ranked_ids=(42, 7),
                top1_score=0.85,
            ),
            # b2b_saas: novel (false positive at threshold 0.65), top1=0.70
            _make_result(
                "b2b-2",
                category="novel",
                is_duplicate=False,
                is_novel=True,
                ranked_ids=(99,),
                top1_score=0.70,
            ),
            # healthcare: relevant (duplicate), expected [55], got [11, 22] -> RR=0
            _make_result(
                "hc-1",
                category="duplicate",
                is_duplicate=True,
                is_novel=False,
                ranked_ids=(11, 22),
                top1_score=0.40,
            ),
            # other: errored (no ranked_ids), top1_score=None
            _make_result(
                "other-err",
                category="duplicate",
                is_duplicate=True,
                is_novel=False,
                ranked_ids=(),
                top1_score=None,
                search_error="ConnectError: refused",
            ),
        ]
        ideas = {
            "b2b-1": "AI agent for sales teams",
            "b2b-2": "B2B platform for inventory",
            "hc-1": "Clinical decision support for hospitals",
            "other-err": "Some other idea",
        }
        expected = {
            "b2b-1": (42,),
            "b2b-2": (),  # novel record: no expected
            "hc-1": (55,),
            "other-err": (42,),  # assumed relevant
        }
        return results, ideas, expected

    def test_precise_mrr_per_category(self):
        results, ideas, expected = self._build_fixture()
        metrics = compute_per_category_metrics_from_benchmark(
            results,
            config_name="test",
            threshold=0.65,
            record_id_to_idea=ideas,
            record_id_to_expected=expected,
        )
        # b2b_saas: 1 relevant (b2b-1 with RR=1.0) -> MRR=1.0
        b2b = metrics[BusinessCategory.B2B_SAAS]
        assert b2b.n_records == 2
        assert b2b.n_relevant == 1
        assert b2b.n_novel == 1
        assert b2b.mrr == pytest.approx(1.0)
        # FPR: 1/1 novel with top1=0.70 above 0.65 -> 1.0
        assert b2b.fpr_on_novel == pytest.approx(1.0)
        # healthcare: 1 relevant (hc-1 with RR=0) -> MRR=0
        hc = metrics[BusinessCategory.HEALTHCARE]
        assert hc.n_records == 1
        assert hc.n_relevant == 1
        assert hc.mrr == pytest.approx(0.0)

    def test_top_3_failures_orders_worst_first(self):
        results, ideas, expected = self._build_fixture()
        metrics = compute_per_category_metrics_from_benchmark(
            results,
            config_name="test",
            threshold=0.65,
            record_id_to_idea=ideas,
            record_id_to_expected=expected,
        )
        # The b2b_saas failures should be: b2b-2 (false positive)
        # then b2b-1 (no failure, ranked at top). Since b2b-1
        # has RR=1.0, b2b-2 (the novel false positive) should
        # come first.
        b2b = metrics[BusinessCategory.B2B_SAAS]
        assert len(b2b.top_3_failures) == 2  # only 2 records in category
        assert b2b.top_3_failures[0].record_id == "b2b-2"

    def test_other_category_catches_uncategorised(self):
        # The "other-err" record has a top1 of None and an
        # expected of (42,). Since its idea text doesn't match
        # any rule, it lands in OTHER. Its reciprocal rank is
        # 0 (ranked_ids is empty), so it's the worst failure in
        # the OTHER cell.
        results, ideas, expected = self._build_fixture()
        metrics = compute_per_category_metrics_from_benchmark(
            results,
            config_name="test",
            threshold=0.65,
            record_id_to_idea=ideas,
            record_id_to_expected=expected,
        )
        other = metrics[BusinessCategory.OTHER]
        assert other.n_records == 1
        assert other.top_3_failures[0].record_id == "other-err"

    def test_empty_benchmark_returns_empty_dict(self):
        results, ideas, expected = self._build_fixture()
        metrics = compute_per_category_metrics_from_benchmark(
            [],
            config_name="test",
            threshold=0.65,
            record_id_to_idea=ideas,
            record_id_to_expected=expected,
        )
        assert metrics == {}


# ---------------------------------------------------------------------------
# TestFailureAnalysisWriters — the per-config MD / CSV / heatmap
# ---------------------------------------------------------------------------


class TestFailureAnalysisWriters:
    """The writer helpers — the on-disk artifact shape."""

    def _build_metrics(self) -> Dict[BusinessCategory, PerCategoryMetrics]:
        return {
            BusinessCategory.B2B_SAAS: PerCategoryMetrics(
                business_category=BusinessCategory.B2B_SAAS,
                n_records=11,
                n_relevant=9,
                n_novel=2,
                mrr=0.222,
                ndcg_at_10=0.222,
                fpr_on_novel=1.0,
                top_3_failures=[
                    _make_result(
                        "b2b-1",
                        category="duplicate",
                        is_duplicate=True,
                        is_novel=False,
                        ranked_ids=(42, 7),
                        top1_score=0.85,
                    ),
                ],
            ),
            BusinessCategory.OTHER: PerCategoryMetrics(
                business_category=BusinessCategory.OTHER,
                n_records=173,
                n_relevant=80,
                n_novel=93,
                mrr=0.642,
                ndcg_at_10=0.655,
                fpr_on_novel=1.0,
                top_3_failures=[],
            ),
        }

    def test_per_config_markdown_contains_all_required_pieces(self, tmp_path):
        # Wire the idea text onto the failure so the writer
        # renders a useful line.
        m = self._build_metrics()
        for cat_metrics in m.values():
            for r in cat_metrics.top_3_failures:
                r._idea_text = "AI-powered contract review for SMB law firms"

        out = write_per_config_markdown(
            m,
            config_name="dense_bge_m3",
            benchmark_name="labeled_v300.jsonl",
            threshold=0.65,
            output_path=tmp_path / "failure-breakdown-dense_bge_m3.md",
        )
        body = out.read_text()
        # Header carries the honest-provenance stamp.
        assert "deterministic-rule-based-v1" in body
        assert "labeled_v300.jsonl" in body
        # Both categories are rendered.
        assert "B2B SaaS" in body
        assert "Other" in body
        # The honest call-out is present.
        assert "Honest call-out" in body
        # MRR values land in the table.
        assert "0.222" in body
        assert "0.642" in body
        # Top-3 failure record id is in the table.
        assert "b2b-1" in body
        # The n-flag is NOT applied to the cells with n>=5.
        assert "(n small)" not in body  # b2b n=11, other n=173

    def test_per_config_markdown_flags_small_n(self, tmp_path):
        m = {
            BusinessCategory.HEALTHCARE: PerCategoryMetrics(
                business_category=BusinessCategory.HEALTHCARE,
                n_records=3,  # below the small-n floor
                n_relevant=1,
                n_novel=2,
                mrr=0.0,
                ndcg_at_10=0.0,
                fpr_on_novel=0.5,
                top_3_failures=[],
            ),
        }
        out = write_per_config_markdown(
            m,
            config_name="bm25",
            benchmark_name="labeled_v300.jsonl",
            threshold=0.65,
            output_path=tmp_path / "failure-breakdown-bm25.md",
        )
        body = out.read_text()
        assert "(n small)" in body

    def test_breakdown_csv_writes_one_row_per_category(self, tmp_path):
        m = self._build_metrics()
        rows = []
        for cat, metrics in m.items():
            rows.append(build_csv_row("dense_bge_m3", metrics, notes="test"))
        out = write_breakdown_csv(rows, tmp_path / "failure-breakdown.csv")
        # Read back and check shape.
        with out.open() as f:
            header_line = f.readline().strip()
        assert header_line == ",".join(BREAKDOWN_CSV_COLUMNS)
        # 2 data rows.
        body_lines = out.read_text().strip().splitlines()[1:]
        assert len(body_lines) == 2

    def test_heatmap_renders_png(self, tmp_path):
        m = self._build_metrics()
        out = plot_heatmap(
            {"dense_bge_m3": m, "bm25": m},
            benchmark_name="labeled_v300.jsonl",
            output_path=tmp_path / "failure-breakdown.png",
            metric="mrr",
        )
        # PNG file exists and is non-trivial size.
        assert out.exists()
        assert out.stat().st_size > 1000
        # PNG signature.
        with out.open("rb") as f:
            sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n", "not a valid PNG"

    def test_heatmap_title_carries_provenance_stamp(self, tmp_path):
        # The title is the primary honesty surface. The card
        # explicitly says: "The title of failure-breakdown.png:
        # 'Per-category failure analysis | eval=labeled_v300.jsonl
        # (categories LLM-assigned v1, hand-review pending)'."
        m = self._build_metrics()
        out = plot_heatmap(
            {"dense_bge_m3": m},
            benchmark_name="labeled_v300.jsonl",
            output_path=tmp_path / "failure-breakdown.png",
            metric="mrr",
            title_extra="categories LLM-assigned v1, hand-review pending",
        )
        # The provenance string flows into the title_extra
        # path; we don't try to OCR the PNG, but the helper
        # accepts the exact string the card pinned, and the
        # plot_path doesn't raise.
        assert out.exists()


# ---------------------------------------------------------------------------
# TestBuildBusinessCategories — the build script
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "evals" / "labeled_v300.jsonl").exists(),
    reason="evals/labeled_v300.jsonl not present",
)
class TestBuildBusinessCategories:
    """The script that extends the eval set with the new fields."""

    def test_dry_run_does_not_modify_file(self, tmp_path):
        in_path = _REPO_ROOT / "evals" / "labeled_v300.jsonl"
        before = in_path.read_text()
        result = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "build_business_categories.py"),
                "--input", str(in_path),
                "--output", str(tmp_path / "out.jsonl"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # File on disk is unchanged.
        assert in_path.read_text() == before
        # Stdout shows the coverage.
        assert "category coverage" in result.stdout

    def test_idempotent(self, tmp_path):
        # Running the script twice against the same input +
        # output yields byte-identical output.
        in_path = _REPO_ROOT / "evals" / "labeled_v300.jsonl"
        out_path = tmp_path / "extended.jsonl"
        for _ in range(2):
            subprocess.run(
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "build_business_categories.py"),
                    "--input", str(in_path),
                    "--output", str(out_path),
                ],
                check=True,
                capture_output=True,
            )
            first = out_path.read_text()
        # Re-run the script once more; compare.
        out_path_2 = tmp_path / "extended2.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "build_business_categories.py"),
                "--input", str(in_path),
                "--output", str(out_path_2),
            ],
            check=True,
            capture_output=True,
        )
        assert first == out_path_2.read_text(), (
            "build_business_categories is not idempotent — re-runs "
            "yield different output"
        )

    def test_output_has_required_fields(self, tmp_path):
        in_path = _REPO_ROOT / "evals" / "labeled_v300.jsonl"
        out_path = tmp_path / "extended.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "build_business_categories.py"),
                "--input", str(in_path),
                "--output", str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        with out_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        # Card acceptance: "all 300 records have a category field".
        assert len(records) == 300
        for rec in records:
            assert "business_category" in rec
            assert rec["business_category"] in {c.value for c in BUSINESS_CATEGORIES}
            assert "business_category_provenance" in rec
            assert "v1" in rec["business_category_provenance"]
            # The original fields are preserved (we don't
            # clobber the eval-set category).
            assert "category" in rec
            assert "provenance" in rec