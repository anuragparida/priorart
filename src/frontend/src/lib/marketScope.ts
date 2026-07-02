import type { MarketScope, MarketScopeConfidence } from '@/lib/api';

// Market-scope colour tokens (shadcn-aligned; no custom palette).
// Backend enum lives in src/llm/schemas.py MarketScope — these are
// the *only* four values. Add a new branch here when the backend adds
// one; otherwise the fallback covers it.
export const marketScopeConfig: Record<
  MarketScope,
  { label: string; className: string; description: string }
> = {
  wide_open: {
    label: 'Wide open',
    className: 'bg-emerald-600/20 text-emerald-300 border-emerald-600/40',
    description: 'No close incumbents in the YC corpus. Genuine first-mover territory.',
  },
  crowded_but_growing: {
    label: 'Crowded but growing',
    className: 'bg-sky-600/20 text-sky-300 border-sky-600/40',
    description: 'Multiple incumbents, but the market is expanding — room for a sharper wedge.',
  },
  saturated: {
    label: 'Saturated',
    className: 'bg-red-600/20 text-red-300 border-red-600/40',
    description: 'Strong incumbents with established distribution. Hard to differentiate.',
  },
  niche_but_real: {
    label: 'Niche but real',
    className: 'bg-amber-600/20 text-amber-300 border-amber-600/40',
    description: 'Small but durable wedge. Real customers, real revenue, but narrow TAM.',
  },
};

export function marketScopeLabel(scope: MarketScope): string {
  return marketScopeConfig[scope]?.label ?? scope;
}

export function marketScopeClass(scope: MarketScope): string {
  return (
    marketScopeConfig[scope]?.className ??
    'bg-secondary text-secondary-foreground border-border'
  );
}

// ---------------------------------------------------------------------------
// Phase 4.6 — 3-level confidence badge for the new MarketScopeSignal envelope.
// ---------------------------------------------------------------------------
//
// The legacy MarketScope enum (above) is unchanged — the existing direction
// chip keeps working. Phase 4 adds an *additive* confidence tier the
// `market_scope_signal` activity computes:
//
//   - "directional":    LLM synthesis fallback when the corpus + web layer
//                       don't settle the direction deterministically. Same
//                       as the Phase 1.7 stub behaviour — explicit about
//                       what it is.
//   - "evidence_backed": >=1 corpus source AND >=1 web source (SearXNG +
//                        Firecrawl). The 4.4 path populates this tier.
//   - "quantitative":    the 4.2 deterministic direction rules fired and
//                        MarketScopeQuant is fully populated. Strongest
//                        signal we have — `MarketScopeQuant.competitor_count`,
//                        `recent_3y_count`, `saturation_index`,
//                        `growth_rate`, `category_distribution` are all real.
//
// The frontend falls back to "directional" when the envelope is absent
// (Phase 1.7/1.8/1.11/2.x/3.x verdict shape) so the badge stays additive
// and the chip never goes away.

export type ConfidenceTier = MarketScopeConfidence;

export const confidenceConfig: Record<
  ConfidenceTier,
  { label: string; className: string; tooltip: string }
> = {
  directional: {
    label: 'directional',
    className: 'bg-zinc-600/20 text-zinc-300 border-zinc-600/40',
    tooltip:
      'LLM synthesis fallback — corpus + web layer didn\'t settle the ' +
      'direction deterministically. Same confidence as the Phase 1.7 stub.',
  },
  evidence_backed: {
    label: 'evidence-backed',
    className: 'bg-sky-600/20 text-sky-300 border-sky-600/40',
    tooltip:
      'Direction settled with ≥1 corpus source + ≥1 web source. ' +
      'Hover the supporting evidence list below for the top-3 web sources.',
  },
  quantitative: {
    label: 'quantitative',
    className: 'bg-emerald-600/20 text-emerald-300 border-emerald-600/40',
    tooltip:
      'Direction settled by deterministic corpus rules. MarketScopeQuant is ' +
      'fully populated — competitor_count, recent_3y_count, ' +
      'saturation_index, growth_rate, category_distribution are real numbers.',
  },
};

export function confidenceLabel(tier: ConfidenceTier): string {
  return confidenceConfig[tier]?.label ?? tier;
}

export function confidenceClass(tier: ConfidenceTier): string {
  return (
    confidenceConfig[tier]?.className ??
    'bg-secondary text-secondary-foreground border-border'
  );
}

export function confidenceTooltip(tier: ConfidenceTier): string {
  return confidenceConfig[tier]?.tooltip ?? '';
}

// Resolve the effective confidence tier from a verdict. Returns "directional"
// when the 4.1 envelope is absent so the chip is always rendered with a
// meaningful label — the chip is additive on top of the direction chip, never
// a replacement.
export function effectiveConfidence(
  signal:
    | { confidence: ConfidenceTier }
    | null
    | undefined,
): ConfidenceTier {
  return signal?.confidence ?? 'directional';
}