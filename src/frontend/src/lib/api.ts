// Wire types matching src/llm/schemas.py (Phase 1.7) + src/api/analyze.py (Phase 1.8).
// Keep these in sync with the backend Pydantic models — they are the
// public contract that the frontend renders.

export type MarketScope =
  | 'wide_open'
  | 'crowded_but_growing'
  | 'saturated'
  | 'niche_but_real';

export interface CompetitorVerdict {
  company_id: number;
  name: string;
  similarity_axes: string[];
  key_differences: string[];
  likely_failure_modes: string[];
  evidence_links: string[];
  confidence: number; // 0-1
}

export interface IdeaVerdict {
  idea: string;
  top_competitors: CompetitorVerdict[];
  market_scope: MarketScope;
  market_scope_rationale: string;
  supporting_evidence: string[];
}

// The /ideas/analyze endpoint returns 200 + structured error on
// every failure path (no 500s). The four error variants below match
// AnalyzeError in src/api/analyze.py.
export interface AnalyzeErrorBody {
  error:
    | 'no_competitors'
    | 'schema_violation'
    | 'llm_transport'
    | 'llm_unconfigured';
  details?: unknown;
}

export type AnalyzeResult =
  | { ok: true; verdict: IdeaVerdict }
  | { ok: false; error: AnalyzeErrorBody };

// /search wire types (Phase 1.4)
export interface SearchHit {
  id: number;
  name: string;
  description: string;
  similarity: number; // raw cosine, [-1, 1]
  confidence: number; // (sim + 1) / 2, [0, 1]
}

export interface SearchResponse {
  hits: SearchHit[];
  query: string;
  model: string;
  top_k: number;
  corpus_count: number;
}

// Backend base URL. In dev Vite proxies /api -> http://localhost:18001
// (see vite.config.ts). In prod, set VITE_API_BASE_URL at build time.
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api';

export async function analyzeIdea(idea: string, topK = 3): Promise<AnalyzeResult> {
  const res = await fetch(`${API_BASE}/ideas/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ idea, top_k: topK }),
  });

  // Per spec: every /ideas/analyze response is 200 + structured body
  // (IdeaVerdict OR AnalyzeError). A non-200 here means a real server
  // fault (500/502/503) — surface it as llm_transport so the UI
  // can show an honest message.
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = { message: `HTTP ${res.status}` };
    }
    return {
      ok: false,
      error: { error: 'llm_transport', details: { status: res.status, body: detail } },
    };
  }

  const body = (await res.json()) as IdeaVerdict | AnalyzeErrorBody;

  if ('error' in body) {
    return { ok: false, error: body };
  }
  return { ok: true, verdict: body };
}

export async function searchCorpus(
  query: string,
  topK = 20,
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    throw new Error(`/search failed: HTTP ${res.status}`);
  }
  return (await res.json()) as SearchResponse;
}