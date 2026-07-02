"""CI helper scripts for the GitHub Actions eval-regression workflow.

Phase 3.6 (card t_e0f62c2a). The three entry points are:

* :mod:`scripts.ci.run_eval_sweep` — 3-config eval-harness sweep.
* :mod:`scripts.ci.eval_gate`     — read the leaderboard CSV, fail
  the build if MRR or FPR-on-novel cross the regression-detection
  thresholds on ``hybrid_rrf``.
* :mod:`scripts.ci.leaderboard_diff` — render a PR-comment-friendly
  Markdown diff between the base and head leaderboard CSVs.

The package exists so the diff module can ``from scripts.ci.eval_gate
import ...`` without re-implementing the row-coercion logic; the
gate and the diff agree on what "selected" and "mrr" mean.
"""
