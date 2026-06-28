# GitHub repository metadata

> Settings to apply on the first push to GitHub. Documented here so
> the values are version-controlled alongside the code.

## Repository description (one line)

> Startup-idea deduplication against the public YC + Product Hunt + HN corpus, with a reproducible eval harness and a labeled benchmark.

## Website

```
https://priorart.dev
```

(Placeholder — replace with the actual deployment URL when Phase 3 ships.)

## Topics

Apply these 9 topics on the first push, in this order:

```
ai
rag
retrieval
evaluation
mlops
pgvector
startups
competitor-research
pydantic
```

Rationale: `ai` / `rag` / `retrieval` for the topical cluster; `evaluation` /
`mlops` for the eval harness + production-grade positioning; `pgvector` /
`pydantic` for the specific stack; `startups` / `competitor-research` for the
problem domain. Eight is the sweet spot — GitHub allows up to 20 but the
discovery algorithm weights the first 10 highest.

## Features checklist

GitHub's repo "About" panel also has a features checklist. Enable:

- [x] Releases
- [ ] Packages (no container registry needed for Phase 1)
- [ ] Deployments (no deployment target yet)

## Social preview

Upload a 1280×640 PNG to **Settings → Social preview** when the
Phase 3 docs polish lands. For now the leaderboard screenshot in
`docs/assets/leaderboard-v1.png` is the visual identity.

## How to apply these settings

Via the GitHub UI:

> Settings → General → Topics

Or via `gh` (once the repo is created):

```bash
gh repo edit --add-topic ai --add-topic rag --add-topic retrieval \
             --add-topic evaluation --add-topic mlops \
             --add-topic pgvector --add-topic startups \
             --add-topic competitor-research --add-topic pydantic
```