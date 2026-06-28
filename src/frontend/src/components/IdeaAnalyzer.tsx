import { useState, type FormEvent } from 'react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import {
  analyzeIdea,
  type AnalyzeErrorBody,
  type IdeaVerdict,
} from '@/lib/api';
import { marketScopeClass, marketScopeLabel } from '@/lib/marketScope';

// --- Error-to-message helpers -----------------------------------------------

function errorTitle(err: AnalyzeErrorBody): string {
  switch (err.error) {
    case 'no_competitors':
      return 'No similar launches found';
    case 'llm_unconfigured':
      return 'LLM is not configured';
    case 'schema_violation':
      return 'LLM returned a malformed response';
    case 'llm_transport':
      return 'LLM transport error';
  }
}

function errorBody(err: AnalyzeErrorBody): string {
  switch (err.error) {
    case 'no_competitors':
      return (
        'No similar launches found in the YC corpus. ' +
        'This is genuinely novel — or outside the YC market.'
      );
    case 'llm_unconfigured':
      return (
        'The structured-comparison step needs an Anthropic API key. ' +
        'Set ANTHROPIC_API_KEY in the backend env, restart the API, and retry. ' +
        'The /search endpoint is still functional without it.'
      );
    case 'schema_violation':
      return (
        'The LLM returned a response that failed Pydantic validation. ' +
        'Inspect the backend logs for the validation error.'
      );
    case 'llm_transport':
      return (
        'The LLM call failed at the transport layer (timeout, network, 5xx). ' +
        'Retry in a few seconds; if it persists, check Anthropic status.'
      );
  }
}

// --- Components -------------------------------------------------------------

function LoadingPanel() {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 py-8 text-muted-foreground">
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-primary" />
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-primary [animation-delay:150ms]" />
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-primary [animation-delay:300ms]" />
        <span className="ml-2 text-sm">
          Embedding, retrieving, comparing against the YC corpus…
        </span>
      </CardContent>
    </Card>
  );
}

function ErrorPanel({ error }: { error: AnalyzeErrorBody }) {
  return (
    <Card className="border-destructive/40">
      <CardHeader>
        <CardTitle className="text-destructive">{errorTitle(error)}</CardTitle>
        <CardDescription>{errorBody(error)}</CardDescription>
      </CardHeader>
      {error.details != null && (
        <CardContent>
          <pre className="overflow-x-auto rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">
            {JSON.stringify(error.details, null, 2)}
          </pre>
        </CardContent>
      )}
    </Card>
  );
}

function CompetitorCard({ c, rank }: { c: IdeaVerdict['top_competitors'][number]; rank: number }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <span className="text-muted-foreground">#{rank}</span>
              <span>{c.name}</span>
            </CardTitle>
            <CardDescription>
              Confidence {(c.confidence * 100).toFixed(0)}% · id {c.company_id}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {c.similarity_axes.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {c.similarity_axes.map((axis) => (
              <Badge key={axis} variant="secondary">
                {axis}
              </Badge>
            ))}
          </div>
        )}

        {c.key_differences.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Key differences
            </h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {c.key_differences.map((d) => (
                <li key={d}>{d}</li>
              ))}
            </ul>
          </div>
        )}

        {c.likely_failure_modes.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Likely failure modes
            </h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {c.likely_failure_modes.map((d) => (
                <li key={d}>{d}</li>
              ))}
            </ul>
          </div>
        )}

        {c.evidence_links.length > 0 && (
          <div>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Evidence
            </h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {c.evidence_links.map((link) => (
                <li key={link}>
                  <a
                    href={link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline-offset-4 hover:underline break-all"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function VerdictPanel({ verdict }: { verdict: IdeaVerdict }) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle>Market scope</CardTitle>
            <span
              className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${marketScopeClass(verdict.market_scope)}`}
            >
              {marketScopeLabel(verdict.market_scope)}
            </span>
          </div>
          <CardDescription>{verdict.market_scope_rationale}</CardDescription>
        </CardHeader>
        {verdict.supporting_evidence.length > 0 && (
          <CardContent>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Supporting evidence
            </h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {verdict.supporting_evidence.map((link) => (
                <li key={link}>
                  <a
                    href={link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary underline-offset-4 hover:underline break-all"
                  >
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </CardContent>
        )}
      </Card>

      <div className="space-y-3">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Top competitors
        </h3>
        {verdict.top_competitors.map((c, i) => (
          <CompetitorCard key={c.company_id} c={c} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}

export function IdeaAnalyzer() {
  const [idea, setIdea] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<
    | { kind: 'idle' }
    | { kind: 'verdict'; verdict: IdeaVerdict }
    | { kind: 'error'; error: AnalyzeErrorBody }
  >({ kind: 'idle' });

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const trimmed = idea.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setResult({ kind: 'idle' });
    try {
      const r = await analyzeIdea(trimmed, 3);
      if (r.ok) {
        setResult({ kind: 'verdict', verdict: r.verdict });
      } else {
        setResult({ kind: 'error', error: r.error });
      }
    } catch (err) {
      setResult({
        kind: 'error',
        error: {
          error: 'llm_transport',
          details: { message: err instanceof Error ? err.message : String(err) },
        },
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">PriorArt</h1>
        <p className="text-sm text-muted-foreground">
          Paste a startup idea. We&apos;ll find the closest YC launches and ask
          Claude to compare them — market scope, similarity axes, key
          differences, and where they tend to fail.
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-3">
        <Textarea
          value={idea}
          onChange={(e) => setIdea(e.target.value)}
          placeholder="e.g. AI-powered legal contract review for SMB law firms"
          rows={4}
          maxLength={4096}
          disabled={loading}
        />
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {idea.length}/4096 · one LLM call per submit, top-3 competitors
          </span>
          <Button type="submit" disabled={loading || idea.trim().length === 0}>
            {loading ? 'Analyzing…' : 'Analyze'}
          </Button>
        </div>
      </form>

      {loading && <LoadingPanel />}

      {!loading && result.kind === 'verdict' && <VerdictPanel verdict={result.verdict} />}
      {!loading && result.kind === 'error' && <ErrorPanel error={result.error} />}

      {!loading && result.kind === 'idle' && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Results will appear here.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default IdeaAnalyzer;