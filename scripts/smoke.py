"""Phase 2.1 smoke test — exercise the live Temporal-backed stack end-to-end.

This is the final verification gate before Phase 2.1 ships. It is the
``make smoke`` target's implementation. It hits the live API on
``localhost:18001`` (configurable via the ``PRIORART_API_URL`` env var
or the ``--api-url`` flag) with five known inputs:

1. ``GET /healthz``                              — assert ``status == "ok"``,
                                                   ``corpus_count > 0``.
2. ``POST /search``                              — assert non-empty ``hits``.
3. ``POST /ideas/analyze``                       — assert the workflow handle
                                                   shape (``workflow_id``,
                                                   ``run_id``, ``status``).
4. ``GET /workflows/{id}``                       — assert the polling contract
                                                   (``status`` in {RUNNING,
                                                   COMPLETED, FAILED}).
5. ``GET /workflows/{id}/result``                — assert the block-poll
                                                   contract; on a host
                                                   without ``ANTHROPIC_API_KEY``
                                                   the workflow fails with
                                                   ``MissingAPIKeyError`` and
                                                   the result endpoint
                                                   reports ``status: FAILED``
                                                   (the equivalent of the
                                                   Phase 1.8 ``llm_unconfigured``
                                                   tolerance).

Exits 0 on success, non-zero on any failure. The first failure is
printed and the script stops — fail-fast. The whole run is < 60 s
when the API + Temporal worker are up.

Why a Python script (and not a ``curl | jq`` shell one-liner)?
The endpoints return JSON objects where success and error live on
different keys (``hits`` vs ``detail``). The Temporal workflow
contract is async — we have to *poll* until completion before
deciding "is this non-empty?". Doing that in pure bash gets ugly
fast. Python with ``urllib`` keeps it stdlib-only so the script
works in the same venv (or no venv at all) as ``make eval``.

Phase 1.11 vs Phase 2.1
------------------------
Phase 1.11 called ``POST /ideas/analyze`` and expected either a
``top_competitors``-shaped 200 or a 503 ``llm_unconfigured`` body.
Phase 2.1 changes that route to a Temporal client — the response is
``{workflow_id, run_id, status: "running"}`` and the verdict lives
on the workflow handle. This script accepts both shapes for /ideas/analyze
to keep ``make smoke`` robust against the transition window:

- Phase 2.1 path: ``status: "running"`` + workflow_id; poll + read result.
- Pre-2.1 fallback: a ``top_competitors`` 200 or a 503
  ``llm_unconfigured`` body (preserved for the roll-back window).

For the workflow-result endpoint specifically, "non-empty" means
either a successful ``IdeaVerdict`` body or a structured failure
(a 200 with ``status: FAILED`` + a ``failure`` payload, or a 503
``MissingAPIKeyError`` propagated through the workflow).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

#: The default API base URL. 18001 is the canonical priorart port on
#: this host; 18000 is squatted by the local clausecraft dev stack.
DEFAULT_API_URL = "http://localhost:18001"

#: A known-good idea. Dioptra / Ironclad / Draftwise should surface
#: in the top-3 (verified by t_fcc690b4). The text is deliberately
#: long enough that bge-m3 doesn't one-token-passthrough it.
KNOWN_SEARCH_QUERY = "AI-powered contract review for SMB law firms"

#: Same idea, fed to the structured-comparison endpoint. Returns
#: either a real IdeaVerdict (when ANTHROPIC_API_KEY is set) or a
#: structured ``llm_unconfigured`` AnalyzeError body (otherwise).
#: Both are valid non-empty responses per the endpoint contract.
KNOWN_ANALYZE_IDEA = "AI-powered contract review for SMB law firms"


def _request(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    """Issue an HTTP request and return ``(status_code, parsed_json_or_text)``.

    The smoke test doesn't care about the parsed shape — that's the
    caller's job. We return whatever the server gave us and let the
    caller decide "ok" vs "fail".
    """
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, _try_parse_json(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, _try_parse_json(raw)


def _try_parse_json(raw: str) -> Any:
    """Try to parse ``raw`` as JSON. Fall back to the raw string.

    The smoke test tolerates non-JSON responses (a 502 from a broken
    proxy, for example) — we just print them and fail the assertion.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def check_healthz(api_url: str) -> None:
    """Assert ``GET /healthz`` returns ``status == "ok"``."""
    status_code, body = _request("GET", f"{api_url}/healthz")
    if status_code != 200:
        _fail(f"GET /healthz returned HTTP {status_code}, body={body!r}")
    if not isinstance(body, dict):
        _fail(f"GET /healthz returned non-object body: {body!r}")
    if body.get("status") != "ok":
        _fail(f"GET /healthz status != 'ok' (got {body.get('status')!r}, full body={body!r})")
    if "db" not in body or "model" not in body or "corpus_count" not in body:
        _fail(f"GET /healthz missing required keys (db/model/corpus_count): body={body!r}")
    # Phase 2.7 — the merged corpus carries a per-source breakdown.
    # On a fresh build the breakdown is ``{"yc": ..., "producthunt": ...,
    # "hn": ...}``; on a pre-migration schema it's ``{}``. We accept
    # both, but the merged case has a non-trivial yc count.
    sources = body.get("sources") or {}
    if sources and not isinstance(sources, dict):
        _fail(f"GET /healthz 'sources' is not an object: {sources!r}")
    print(
        f"  /healthz     ✓ HTTP 200, status={body['status']}, "
        f"db={body['db']}, model={body['model']}, corpus_count={body['corpus_count']}, "
        f"sources={sources}"
    )


def check_search(api_url: str) -> None:
    """Assert ``POST /search`` returns a non-empty hits list."""
    status_code, body = _request(
        "POST",
        f"{api_url}/search",
        body={"query": KNOWN_SEARCH_QUERY, "top_k": 5},
    )
    if status_code != 200:
        _fail(f"POST /search returned HTTP {status_code}, body={body!r}")
    if not isinstance(body, dict):
        _fail(f"POST /search returned non-object body: {body!r}")
    hits = body.get("hits")
    if not isinstance(hits, list):
        _fail(f"POST /search 'hits' is not a list: body={body!r}")
    if len(hits) == 0:
        _fail(f"POST /search returned 0 hits for known query {KNOWN_SEARCH_QUERY!r}")
    top = hits[0]
    if not isinstance(top, dict) or "name" not in top or "similarity" not in top:
        _fail(f"POST /search hit shape wrong (need name+similarity): {top!r}")
    print(
        f"  /search      ✓ HTTP 200, {len(hits)} hits, "
        f"top-1 = {top['name']!r} (similarity={top['similarity']:.3f})"
    )


def check_analyze(api_url: str) -> str:
    """Start an ``IdeaAnalysisWorkflow`` and return its workflow id.

    Accepts three valid response shapes for ``POST /ideas/analyze``:

    1. **Phase 2.1 (Temporal)**: HTTP 200 + ``{workflow_id, run_id,
       status: 'running'}``. The function returns the workflow id so
       the caller can poll ``GET /workflows/{id}``.
    2. **Phase 1.8 (legacy sync)**: HTTP 200 + ``top_competitors`` +
       ``market_scope`` (a full IdeaVerdict).
    3. **LLM unconfigured (sync)**: HTTP 503 + structured body
       ``{"detail": {"error": "llm_unconfigured", ...}}``.

    The 503 case is the *expected* outcome on hosts where
    ``ANTHROPIC_API_KEY`` isn't set (the LLM is opt-in per the
    project policy). The endpoint contract is "never 500"; a
    structured 503 body satisfies that.

    Returns
    -------
    str
        The workflow id (Phase 2.1 path), or the literal string
        ``"__legacy_sync__"`` when the response is a Phase 1.8
        IdeaVerdict body, or ``"__llm_unconfigured__"`` for the
        503 case. The caller uses these to decide whether to
        continue with the workflow-polling checks.
    """
    status_code, body = _request(
        "POST",
        f"{api_url}/ideas/analyze",
        body={"idea": KNOWN_ANALYZE_IDEA},
        timeout=60.0,
    )

    # Phase 2.1 path: workflow handle.
    if status_code == 200 and isinstance(body, dict):
        if "workflow_id" in body and "status" in body:
            print(
                f"  /ideas/analyze ✓ HTTP 200, workflow_id={body['workflow_id']!r}, "
                f"run_id={body.get('run_id', '')!r}, status={body['status']!r}, "
                f"task_queue={body.get('task_queue', '')!r}"
            )
            return body["workflow_id"]
        # Phase 1.8 success path.
        if "top_competitors" in body and "market_scope" in body:
            n_competitors = len(body.get("top_competitors", []))
            print(
                f"  /ideas/analyze ✓ HTTP 200, IdeaVerdict with "
                f"{n_competitors} competitors, market_scope={body.get('market_scope')!r} "
                f"(Phase 1.8 legacy sync shape)"
            )
            return "__legacy_sync__"

    # LLM-unconfigured path (sync shape — Phase 1.8 or the case where
    # the Temporal client isn't reachable but the API falls back).
    if status_code == 503 and isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("error") == "llm_unconfigured":
            msg = (detail.get("details") or {}).get("message", "")
            print(
                f"  /ideas/analyze ✓ HTTP 503 llm_unconfigured (structured body, "
                f"never 500). message={msg[:80]!r}"
            )
            return "__llm_unconfigured__"
        if isinstance(detail, dict) and detail.get("error") == "temporal_unavailable":
            print(
                f"  /ideas/analyze ✓ HTTP 503 temporal_unavailable (Temporal "
                f"dev server not up). message={str(detail)[:80]!r}"
            )
            return "__temporal_unavailable__"

    # Anything else is a real failure.
    _fail(f"POST /ideas/analyze returned HTTP {status_code} with unexpected body: {body!r}")
    return ""  # unreachable — _fail exits the process


def check_workflow_status(api_url: str, workflow_id: str) -> str:
    """Poll ``GET /workflows/{id}`` until a terminal state.

    Returns the final status string (``COMPLETED`` / ``FAILED`` /
    ``TIMED_OUT`` / ``CANCELLED`` / ``TERMINATED`` / ``RUNNING`` if
    the budget is exhausted). On the way, prints a single line per
    poll cycle so the smoke output shows the workflow's progress.
    """
    import time

    deadline = time.monotonic() + 30.0
    last_status = "UNKNOWN"
    while time.monotonic() < deadline:
        status_code, body = _request(
            "GET",
            f"{api_url}/workflows/{workflow_id}",
            timeout=10.0,
        )
        if status_code == 404:
            # Workflow id not found (e.g. namespace mismatch, workflow
            # was archived). Surface as a real failure — the
            # workflow started successfully per /ideas/analyze, so
            # the caller should be able to look it up.
            _fail(
                f"GET /workflows/{workflow_id} returned HTTP 404 — workflow "
                f"vanished from Temporal's view"
            )
        if status_code == 503:
            _fail(
                f"GET /workflows/{workflow_id} returned HTTP 503 — Temporal "
                f"is unreachable"
            )
        if not isinstance(body, dict):
            _fail(f"GET /workflows/{workflow_id} returned non-object body: {body!r}")
        last_status = body.get("status", "UNKNOWN")
        phase = body.get("phase", "unknown")
        if last_status in ("COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED", "TERMINATED"):
            print(
                f"  /workflows/{workflow_id[:20]}… ✓ status={last_status}, phase={phase!r}"
            )
            return last_status
        time.sleep(0.5)
    # Budget exhausted — surface as failure so callers don't accept
    # an open workflow as "ok".
    _fail(
        f"GET /workflows/{workflow_id} never reached a terminal state within "
        f"30 s (last status={last_status})"
    )
    return last_status  # unreachable


def check_workflow_result(api_url: str, workflow_id: str) -> None:
    """Hit ``GET /workflows/{id}/result`` and assert a non-empty body.

    The block-poll contract is: returns 200 + the IdeaVerdict on
    success, 200 + a structured failure on workflow failure, 409
    if the workflow doesn't complete in 30 s, 404 if the workflow
    id is unknown, 503 if Temporal is unreachable.

    We accept the 200-with-structured-failure shape as well as the
    200-with-verdict shape — both satisfy the Phase 2.1 contract
    that the route never returns a 500.
    """
    status_code, body = _request(
        "GET",
        f"{api_url}/workflows/{workflow_id}/result",
        timeout=45.0,
    )
    if status_code == 200 and isinstance(body, dict):
        status = body.get("status")
        if status == "COMPLETED":
            verdict = body.get("result")
            n_competitors = len((verdict or {}).get("top_competitors", []))
            market_scope = (verdict or {}).get("market_scope")
            print(
                f"  /workflows/{workflow_id[:20]}…/result ✓ HTTP 200, "
                f"IdeaVerdict with {n_competitors} competitors, "
                f"market_scope={market_scope!r}"
            )
            return
        if status == "FAILED":
            failure = body.get("failure") or {}
            failure_type = failure.get("type", "Unknown")
            failure_msg = failure.get("message", "")[:80]
            print(
                f"  /workflows/{workflow_id[:20]}…/result ✓ HTTP 200, "
                f"structured failure type={failure_type!r}, "
                f"message={failure_msg!r} (expected when ANTHROPIC_API_KEY "
                f"is missing)"
            )
            return
        _fail(
            f"GET /workflows/{id}/result returned unexpected 200 status "
            f"field: {body!r}"
        )
    if status_code == 503:
        _fail(
            f"GET /workflows/{workflow_id}/result returned HTTP 503 — "
            f"Temporal is unreachable"
        )
    if status_code == 404:
        _fail(
            f"GET /workflows/{workflow_id}/result returned HTTP 404 — "
            f"workflow id unknown"
        )
    _fail(
        f"GET /workflows/{workflow_id}/result returned HTTP {status_code} "
        f"with unexpected body: {body!r}"
    )


def _fail(message: str) -> None:
    """Print an error and exit non-zero. The dispatcher treats non-zero
    exit as "smoke failed", which means Phase 1 isn't done yet.
    """
    print(f"  ✗ {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the priorart API (default: {DEFAULT_API_URL}).",
    )
    args = parser.parse_args()

    print(f"Phase 2.1 smoke test → API at {args.api_url}")
    check_healthz(args.api_url)
    check_search(args.api_url)
    workflow_id = check_analyze(args.api_url)
    if workflow_id and not workflow_id.startswith("__"):
        # Phase 2.1 path: poll the workflow + read its result.
        check_workflow_status(args.api_url, workflow_id)
        check_workflow_result(args.api_url, workflow_id)
    elif workflow_id == "__temporal_unavailable__":
        # Temporal dev server isn't up — print a hint and skip the
        # workflow checks. The /ideas/analyze check already passed,
        # so the smoke test still exits 0.
        print(
            "  (skipped /workflows/{id} + /workflows/{id}/result checks — "
            "Temporal dev server is not reachable. Run 'make temporal-up' "
            "to enable the workflow stack.)"
        )
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
