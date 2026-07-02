// Wire types matching src/llm/schemas.py (Phase 1.7 + Phase 4.1) +
// src/api/analyze.py (Phase 1.8). Keep these in sync with the
// backend Pydantic models — they are the public contract that the
// frontend renders.

export type MarketScope =
  | 'wide_open'
  | 'crowded_but_growing'
  | 'saturated'
  | 'niche_but_real';

// Phase 4.1 — confidence tier for the market-scope signal envelope.
// Mirrors MarketScopeConfidence in src/llm/schemas.py. The frontend
// renders a small badge next to the direction chip; the badge falls
// back to 'directional' when `market_scope_signal` is null (the
// Phase 1.7/1.8/1.11/2.1/2.2/2.3/2.8/3.1/3.3 verdict shape).
export type MarketScopeConfidence =
  | 'directional'
  | 'evidence_backed'
  | 'quantitative';

// Phase 4.1 — one evidence entry. The LLM synthesis step in 4.4
// cites evidence but never invents it; `corpus` carries a company_id,
// `web` carries a url. as_of is the ISO-8601 capture timestamp.
export interface MarketScopeEvidence {
  source: 'corpus' | 'web';
  url?: string | null;
  company_id?: number | null;
  snippet?: string | null;
  as_of: string;
}

// Phase 4.1 — quantitative layer populated by the 4.2/4.3
// deterministic rules. Null when confidence is 'directional';
// partially populated (search_volume_proxy set) when 'evidence_backed'.
export interface MarketScopeQuant {
  competitor_count: number;
  recent_3y_count: number;
  category_distribution: Record<string, number>;
  search_volume_proxy?: number | null;
  saturation_index: number; // [0, 1]
  growth_rate?: number | null;
}

// Phase 4.1 — the corpus-grounded envelope. Optional on the
// verdict; null when the 4.2 deterministic rules don't fire and
// the pipeline falls back to the Phase 1.7 stub shape.
export interface MarketScopeSignal {
  direction: MarketScope;
  rationale: string;
  quantitative?: MarketScopeQuant | null;
  confidence: MarketScopeConfidence;
  evidence: MarketScopeEvidence[];
}

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
  // Phase 4.1 — additive. The legacy `market_scope` + `market_scope_rationale`
  // fields are unchanged for backward compat; the badge in 4.6 reads
  // `market_scope_signal.confidence` when present and falls back to
  // 'directional' when null.
  market_scope_signal?: MarketScopeSignal | null;
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