#!/usr/bin/env bash
# scripts/demo.sh — 2-minute PriorArt end-to-end demo (recorded by asciinema).
#
# What this shows (in order):
#   1. The eval harness runs against a 25-record stratified slice of
#      labeled_v300.jsonl (~15-25s) and writes the leaderboard markdown.
#      The full 300-record run takes ~75s and is what `make eval` does by
#      default; the truncated slice keeps the demo inside the 2-minute
#      budget while still exercising the same code path.
#   2. The /search REST endpoint, called with the demo idea.
#      Shows the ranked YC + Product Hunt + HN competitors.
#   3. POST /ideas/analyze — starts a Temporal workflow. We poll the
#      result endpoint until the workflow reaches a terminal state.
#      If ANTHROPIC_API_KEY is unset, the workflow fails at the LLM
#      step with a structured ApplicationError — that's the real
#      production failure mode (missing key) and we show it honestly
#      rather than papering over it.
#   4. The cumulative leaderboard.csv tail (real numbers from the
#      Phase 2.9/3.3 runs on labeled_v300.jsonl).
#   5. The web UIs the operator can open after the demo.
#
# This script is for demo recording only. It is NOT a CI / regression
# script. Do not run it as part of `make smoke` or any test workflow.
# Cohere rerank stays out of the demo on purpose (PHASE-3.md §Pitfall).

set -euo pipefail

# Resolve repo root from this script's location so the demo is runnable
# from anywhere (asciinema invokes it via `bash scripts/demo.sh`).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Python env — the same one that already serves the API and the worker
# on this host. Keeps the demo dependency-free.
export PYTHONPATH="/tmp/priorart-venv/lib/python3.12/site-packages:${REPO_ROOT}/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
PY="${PY:-/usr/bin/python3.12}"

API_PORT="${API_PORT:-18001}"
API_BASE="http://localhost:${API_PORT}"
DEMO_IDEA='AI-powered contract review for SMB law firms.'
DEMO_BENCH="/tmp/priorart-demo-bench.jsonl"
DEMO_OUT="/tmp/priorart-demo-leaderboard"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://localhost:15000}"

section() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
note()    { printf '\033[2m%s\033[0m\n' "$*"; }
pause()   { sleep "${1:-1}"; }

clear
cat <<'BANNER'
  ____           _    ___          _
 |  _ \ _ __ ___/ \  |_ _|_ __ ___| |_ ___
 | |_) | '__/ _ \ |   | || '__/ _ \ __/ _ \
 |  __/| | | (_) | |  | || | |  __/ ||  __/
 |_|   |_|  \___/|_| |___|_|  \___|\__\___|

  startup-idea dedup — 2-minute demo
BANNER

# ---------------------------------------------------------------------------
# Step 1 — eval harness on a 25-record stratified slice.
# ---------------------------------------------------------------------------
section "1. Eval harness — 25-record stratified slice of labeled_v300.jsonl"

note "Stratified sample: 15 duplicates + 10 novel. Full 300-record run is ~75s."
note "The slice keeps the demo under 2 minutes; the leaderboard.csv tail below has"
note "the real numbers from the full Phase 2.9 / 3.3 / 3.5 runs."
pause 2

"$PY" - <<PYEOF
import json, random
random.seed(11)
with open("${REPO_ROOT}/evals/labeled_v300.jsonl") as f:
    records = [json.loads(l) for l in f if l.strip()]
dup = [r for r in records if r.get("is_duplicate")]
nov = [r for r in records if not r.get("is_duplicate")]
sample = dup[:15] + nov[:10]
random.shuffle(sample)
with open("${DEMO_BENCH}", "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(sample)} records to ${DEMO_BENCH} (15 dup + 10 novel)")
PYEOF

note "Running eval (dense_bge_m3 config, ~10-15s on the 25-record slice)..."
MLFLOW_TRACKING_URI="${MLFLOW_URI}" "$PY" -m eval.run \
    --benchmark "${DEMO_BENCH}" \
    --config "${REPO_ROOT}/configs/dense_bge_m3.yaml" \
    --output "${DEMO_OUT}.csv" \
    --db none \
    --markdown-out "${DEMO_OUT}.md" \
    --mlflow-tracking-uri "${MLFLOW_URI}" \
    --experiment-name "priorart-demo-recording" 2>&1 \
    | grep -E '^\[eval\] config=|^\[eval\] ece=|^\| \*\*|^Best threshold|^[eval].*done in|search_errors=|^🏃|^🧪|^[mlflow]' \
    | sed -E 's/Records in calibration/Records_in_calibration/'

pause 3

# ---------------------------------------------------------------------------
# Step 2 — POST /search with the demo idea.
# ---------------------------------------------------------------------------
section "2. POST /search — ranked competitors for the demo idea"

printf 'idea: %s\n\n' "${DEMO_IDEA}"
pause 2

curl -fsS "${API_BASE}/search" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"query":"%s","top_k":5,"mode":"hybrid"}' "${DEMO_IDEA}")" \
    | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
mode = d.get('mode', 'dense')
print(f\"  corpus_count={d.get('corpus_count')}  mode={mode}  top_k={d.get('top_k')}\")
print()
if mode == 'hybrid':
    print('  (similarity is the RRF fused score; confidence is the dense-path confidence)')
elif mode == 'bm25':
    print('  (similarity is the BM25 score; confidence is the dense-path confidence for the same hit)')
print()
for i, h in enumerate(d.get('hits', []), 1):
    name = h.get('name', '')[:40]
    sim = h.get('similarity', 0)
    conf = h.get('confidence', 0)
    desc = (h.get('description') or '')[:60]
    print(f'  {i}. {name:<40} sim={sim:.3f}  conf={conf:.3f}')
    print(f'     \"{desc}...\"')
    print()
"
pause 3

# ---------------------------------------------------------------------------
# Step 3 — POST /ideas/analyze → Temporal workflow → poll result.
# ---------------------------------------------------------------------------
section "3. POST /ideas/analyze — Temporal workflow (IdeaAnalysisWorkflow)"

WORKFLOW_JSON=$(curl -fsS "${API_BASE}/ideas/analyze" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"idea":"%s","top_k":5}' "${DEMO_IDEA}")")
echo "  workflow handle: ${WORKFLOW_JSON}"

WORKFLOW_ID=$(echo "${WORKFLOW_JSON}" | "$PY" -c "import json,sys; print(json.load(sys.stdin)['workflow_id'])")
printf '\n  workflow_id: %s\n' "${WORKFLOW_ID}"
pause 2

note "Polling /workflows/{id} — terminates when the workflow reaches a terminal state."
# Poll every 2s up to 8s (covers the typical ~3-5s Temporal LLM-fail path).
# We deliberately don't poll for 30s — the demo is capped at ~2 minutes.
TERMINAL=""
for i in 1 2 3 4; do
    sleep 2
    RESP=$(curl -fsS "${API_BASE}/workflows/${WORKFLOW_ID}" 2>/dev/null || echo "")
    if [ -z "${RESP}" ]; then
        printf '  [poll %d/4] (no response)\n' "$i"
        continue
    fi
    LINE=$(echo "${RESP}" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
status = d.get('status', '?')
phase = d.get('phase', '')
print(f'{status:>10}  phase={phase}')
" 2>/dev/null)
    printf '  [poll %d/4] %s\n' "$i" "${LINE}"
    case "${LINE}" in
        *COMPLETED*|*FAILED*|*TIMED_OUT*|*CANCELLED*|*TERMINATED*)
            TERMINAL="yes"
            break
            ;;
    esac
done
if [ -z "${TERMINAL}" ]; then
    note "workflow did not reach a terminal state in the poll window; final state above."
fi

printf '\n  Final workflow result:\n\n'
curl -fsS "${API_BASE}/workflows/${WORKFLOW_ID}/result" \
    | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
print(f\"  status: {d.get('status')}\")
print(f\"  phase:  {d.get('phase')}\")
if d.get('failure'):
    cause = d['failure'].get('cause', {}).get('cause', {})
    msg = cause.get('message', '(no message)')
    print(f\"  failure.message: {msg[:120]}\")
    print()
    print('  (workflow reached the LLM step, then failed on missing ANTHROPIC_API_KEY.')
    print('   This is a real production failure mode. The Temporal trace shows the')
    print('   activity error and the phase progression; the LLM call was never made.)')
elif d.get('result'):
    r = d['result']
    print(f\"  verdict.market_scope: {r.get('market_scope')}\")
    print(f\"  verdict.confidence:   {r.get('confidence')}\")
    print(f\"  verdict.summary:      {(r.get('summary') or '')[:120]}...\")
"
pause 3

# ---------------------------------------------------------------------------
# Step 4 — cumulative leaderboard.csv (Phase 2.9 / 3.3 / 3.5 runs).
# ---------------------------------------------------------------------------
section "4. Cumulative leaderboard.csv — real runs on labeled_v300.jsonl"

note "Showing the most recent run per (config, benchmark) pair."
"$PY" - <<'PYEOF'
import csv
from collections import defaultdict
latest = {}
with open("results/leaderboard.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get("selected_threshold", "").lower() != "true":
            continue
        key = (row["config"], row["benchmark"])
        # Keep the latest (last in file order is fine — append mode)
        latest[key] = row
for (config, bench), row in sorted(latest.items()):
    print(f"  {config:14} {bench:22} "
          f"mrr={float(row['mrr']):.3f}  "
          f"fpr={float(row['fpr_on_novel']):.3f}  "
          f"ece={float(row['ece']):.3f}  "
          f"novel_set_mrr={float(row['novel_set_mrr']):.3f}")
PYEOF
pause 3

# ---------------------------------------------------------------------------
# Step 5 — web UIs the operator can open after the demo.
# ---------------------------------------------------------------------------
section "5. Operator UIs (open in a browser after the demo)"

printf '  Frontend (Vite/React):   %s\n' "http://localhost:15174"
printf '  Dagster UI:              %s\n' "http://localhost:13002"
printf '  MLflow UI:               %s\n' "http://localhost:15000"
printf '  Temporal Web UI:         %s\n' "http://localhost:8233"
pause 2

section "6. Recap + how to re-run with a real LLM verdict"
cat <<'EOF'
  What you just watched (recap):
    eval.run       - POST /search N times (25 in the demo slice,
                     300 in the full Phase 2.9 / 3.3 / 3.5 runs),
                     append to leaderboard.csv, log to MLflow
    /search        - single-shot ANN retrieval (bge-m3 + HNSW,
                     three modes: dense / bm25 / hybrid)
    /ideas/analyze - start IdeaAnalysisWorkflow on Temporal
                     (search -> compare(LLM) -> market_scope(LLM) -> result)
    leaderboard.csv - cumulative experiment history, queryable via DuckDB

  The /ideas/analyze step failed on the LLM call because no Anthropic key
  is configured on this host. To run the full verdict end-to-end:
      echo "sk-ant-..." > ~/.anthropic_key
      pkill -f src.workflow.worker; make temporal-down; make temporal-up
      make worker &
      bash scripts/demo.sh                     # re-record

  Open this cast:  asciinema play docs/assets/demo.cast
EOF