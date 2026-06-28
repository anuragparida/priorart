import type { MarketScope } from '@/lib/api';

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