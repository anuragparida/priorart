"""Phase 1.11 smoke test — exercise the three shipped endpoints end-to-end.

This is the final verification gate before Phase 1 ships. It is the
``make smoke`` target's implementation. It hits the live API on
``localhost:18001`` (configurable via the ``PRIORART_API_URL`` env var
or the ``--api-url`` flag) with three known inputs:

1. ``GET /healthz``            — assert ``status == "ok"``.
2. ``POST /search``            — assert non-empty ``hits`` list.
3. ``POST /ideas/analyze``     — assert the response body is non-empty
                                 and well-formed. A 503 with a structured
                                 ``llm_unconfigured`` body is accepted
                                 (the LLM is opt-in; the endpoint
                                 contract is "never return 500").

Exits 0 on success, non-zero on any failure. The first failure is
printed and the script stops — fail-fast. The whole run is < 5
seconds when the API is up, so it is cheap to wire into CI as the
Phase 1 regression gate.

Why a Python script (and not a ``curl | jq`` shell one-liner)?
The endpoints return JSON objects where success and error live on
different keys (``hits`` vs ``detail``). The ``/ideas/analyze`` 503
response is a structured body with the same shape as the success
body, so the smoke check has to *parse* the JSON before deciding
"is this non-empty?". Doing that in pure bash gets ugly fast.
Python with ``urllib`` keeps it stdlib-only so the script works in
the same venv (or no venv at all) as ``make eval``.
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
    print(
        f"  /healthz     ✓ HTTP 200, status={body['status']}, "
        f"db={body['db']}, model={body['model']}, corpus_count={body['corpus_count']}"
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


def check_analyze(api_url: str) -> None:
    """Assert ``POST /ideas/analyze`` returns a non-empty body.

    Accepts two valid response shapes:

    1. **Success**: HTTP 200 + a Pydantic IdeaVerdict (carries
       ``top_competitors`` + ``market_scope``).
    2. **LLM unconfigured**: HTTP 503 + structured body
       ``{"detail": {"error": "llm_unconfigured", ...}}``.

    The 503-with-structured-body case is the *expected* outcome on
    hosts where ``ANTHROPIC_API_KEY`` isn't set (the LLM is opt-in
    per the project policy). The endpoint contract is "never 500";
    a structured 503 body satisfies that.
    """
    status_code, body = _request(
        "POST",
        f"{api_url}/ideas/analyze",
        body={"idea": KNOWN_ANALYZE_IDEA},
        timeout=60.0,
    )
    # Success path
    if status_code == 200 and isinstance(body, dict):
        if "top_competitors" not in body or "market_scope" not in body:
            _fail(f"POST /ideas/analyze 200 missing top_competitors/market_scope: {body!r}")
        n_competitors = len(body.get("top_competitors", []))
        print(
            f"  /ideas/analyze ✓ HTTP 200, IdeaVerdict with "
            f"{n_competitors} competitors, market_scope={body.get('market_scope')!r}"
        )
        return

    # LLM-unconfigured path (expected when ANTHROPIC_API_KEY is missing)
    if status_code == 503 and isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and detail.get("error") == "llm_unconfigured":
            msg = (detail.get("details") or {}).get("message", "")
            print(
                f"  /ideas/analyze ✓ HTTP 503 llm_unconfigured (structured body, "
                f"never 500). message={msg[:80]!r}"
            )
            return
        _fail(f"POST /ideas/analyze 503 with unexpected detail shape: {body!r}")

    # Anything else is a real failure.
    _fail(f"POST /ideas/analyze returned HTTP {status_code} with unexpected body: {body!r}")


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

    print(f"Phase 1.11 smoke test → API at {args.api_url}")
    check_healthz(args.api_url)
    check_search(args.api_url)
    check_analyze(args.api_url)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
