# Landscape

> The competitive landscape for "paste your idea, get a competitor analysis." The honest lay of the land — what already exists, what it does well, what it doesn't do. This is the "no public tool does all of this" argument that justifies the project.

---

## TL;DR

Three categories of existing tools, each missing something we have:

1. **AI-wrapper "validate your idea" tools** (Siftt, IdeasGPT, ValidatorAI, Sprintbase) — fast UX, but thin LLM + Google search. No curated corpus, no retrieval quality measured, no structured comparison, no eval harness. Vibes, not evidence.
2. **Market-intelligence platforms** (Crunchbase Pro, Pitchbook, CB Insights, SEMrush, Ahrefs, SimilarWeb) — real data, investor-grade, paywalled at the level you need. The opposite of a demo project.
3. **Internal accelerator tooling** (YC, a16z, Antler, Techstars) — the actual production version. Processes 10K+ applications per cycle. Locked behind NDAs.

The academic work on startup success (Gornall, Huang et al.) has the data but no productionized system. The eval libraries (DeepEval, RAGAS, TruLens) are generic — they don't have a domain-specific idea-dedup benchmark.

**The gap:** no public tool does **idea → vector dedup against a labeled public corpus → structured LLM comparison → market-scope signal → reproducible eval harness**, end-to-end. That's our hole.

---

## The full table

| Tool | What it does | What it does well | What it doesn't do | Pricing |
|---|---|---|---|---|
| **Siftt** (siftt.ai) | "Validate your startup idea" wrapper. | Fast onboarding, decent LLM prompt engineering. | Thin wrapper. No curated corpus — just Google search + LLM summary. No retrieval quality measured. No eval harness. | Freemium |
| **IdeasGPT** | Same genre. Same UX pattern. | Cheap, accessible. | Same as Sifft — no corpus, no measurement, vibes. | Free / paid tiers |
| **ValidatorAI** | "Get feedback on your idea" — LLM-as-judge pattern. | One-click UX. | LLM judge with no ground-truth reference. No retrieval. | Freemium |
| **Sprintbase** | Idea validation + lean canvas generator. | Slightly more structured output (lean canvas, SWOT). | Still LLM-with-search, no corpus, no measurement. | Free |
| **Glimpse** | Trend detection, niche discovery. | Good at surfacing growing search terms. | Adjacent to market-scope signal, not duplicate detection. Different problem. | Paid |
| **Exploding Topics** | Same as Glimpse. Trend-focused. | Decent for "is this category growing?" | Not for "does this exact thing exist?" | Paid |
| **SparkToro** | Audience intelligence. | Great for "who talks about X." | Not for competitor dedup. Different problem. | Paid |
| **Crunchbase Pro** | Company + funding database. | Real data, real funding rounds, real investors. | $10K+/yr. Investor-facing, not founder-facing. No "drop your idea in" UX. | $10K+/yr |
| **Pitchbook** | Same genre, more institutional. | Real data. | Even more expensive. Closed corpus. | $20K+/yr |
| **CB Insights** | Market intelligence + predictive analytics. | Good for "is this market growing?" | Enterprise SaaS pricing. Closed. Not founder-facing. | Enterprise |
| **SEMrush** | SEO + traffic estimation. | Real traffic data, real keyword volumes. | Paywalled at the level you need. Free tier is throttled. | $130+/mo |
| **Ahrefs** | Same as SEMrush. | Slightly better backlink data. | Same paywall story. | $100+/mo |
| **SimilarWeb** | Web traffic estimation. | Best free tier of the three. | Free tier is sampled, not real. Real data is enterprise-priced. | Freemium → $30K+/yr |
| **Gornall, Huang et al. (academic)** | Empirical work on startup success, "P(roduct) Market Fit," founder team composition. | Real data, real findings. | No productionized system. The data is here; the tool isn't. | Papers |
| **Startup Graveyard** (startupgraveyard.io) | Public failure postmortems. | Real "this died, here's why" signal. | Static, not queryable, not embedded into a comparison engine. | Free |
| **Autopsy.io** | Same genre, postmortem-style. | Same. | Same. | Free |
| **Internal YC / a16z / Antler / Techstars tooling** | The real production version. Processes 10K+ applications per cycle. | Exactly the right shape. | Locked behind NDAs. No public version, no eval harness, no paper. | N/A |
| **DeepEval** (library) | Generic LLM eval library. | Industry-standard. | Generic — doesn't have a labeled *idea-dedup* benchmark. You'd use it as a layer, not a replacement. | Open source |
| **RAGAS** (library) | RAG-specific metrics. | Faithfulness, context precision, answer relevance out of the box. | Same as DeepEval — generic. No domain benchmark. | Open source |
| **TruLens** (library) | LLM observability + eval. | Good traces. | Same. | Open source |
| **This project** | End-to-end: idea → vector dedup against a public corpus → structured LLM comparison → market-scope signal → reproducible eval harness → MLOps platform. | The full stack. | A demo, not a SaaS. Self-hosted, single-tenant, public-data only. | N/A |

---

## What this means for the project's positioning

The project does not need to be better than Crunchbase at "real market intel" — it's a demo, not a SaaS. The project needs to be **the only public thing that does the full flow, with a benchmark behind it**. That's the positioning.

When someone says "this is just another AI wrapper," the answer is:

> "Show me the eval harness. Show me the 300-idea labeled benchmark. Show me the regression suite that fails the build when MRR drops. Show me the Temporal workflow that retries on transient LLM failures and parks low-confidence verdicts for human review. Show me the Dagster asset lineage that re-embeds the corpus when a snapshot changes. Show me the calibration curve and the FPR-on-novel breakdown. **No public wrapper has any of this.** That's the differentiator."

The eval harness is the moat. The MLOps platform is the credibility. The product is the demo.

---

## What we explicitly don't try to compete with

- **Crunchbase on funding data.** We don't have it. Don't fake it.
- **SEMrush on traffic estimation.** We don't have it. Stub the market-scope signal and label it directional.
- **Internal accelerator tooling.** We don't have the NDA-protected datasets. Public corpus is the limitation. Say it.
- **Glimps / Exploding Topics on trend detection.** Different problem. Adjacent only.

Be honest about what we're not. The "Limitations" section in the README is part of the credibility story.
