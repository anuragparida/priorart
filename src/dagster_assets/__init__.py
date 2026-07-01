"""Dagster integration — Phase 3.1 (card t_7928b3e2).

This package wraps the existing Phase 1.2/1.3/2.5/2.6/2.7 ingestion
scripts as Dagster **assets**. We do not reimplement the scrapers or
the corpus build — those modules are still owned by ``src/data/``.
Dagster only models the lineage and the schedule.

Naming note
-----------
The package is ``src/dagster_assets/`` rather than ``src/dagster/``
because ``from dagster import …`` shadows when a sibling package
named ``dagster`` is on ``sys.path`` (the real ``dagster`` package
is resolved from the venv site-packages, but a project-local
``src/dagster/`` wins because it's closer to the entry point).
``src/dagster_assets/`` avoids the collision.

Design contract (docs/PHASE-3.md §3.1 + docs/ARCHITECTURE.md §Dagster):

    yc_directory ─┐
    product_hunt_archive ─┼─→ company_embeddings ─→ eval_benchmark
    hn_show_posts ─┘              (staleness only)

Plus a ``@daily_schedule`` for the nightly re-embedding and a
``@sensor`` (sensor lands in 3.2 — declared but not built here).

Hard rules (card body):
- Dagster on port 13002, separate container.
- Default retrieval config MUST run offline. Asset materialization
  may hit the scraper sources; everything else is local.
- Cohere rerank stays opt-in only. No 4th config.
- Branch: ``main``. Commit subject includes the card id.
- File ownership: code + ``pyproject.toml`` + ``docker-compose.yml``
  + ``Makefile`` + ``models.yaml``. Don't touch docs.

Why the assets run subprocesses, not Python imports
---------------------------------------------------
The scrapers (``scrape_yc``, ``scrape_ph``, ``scrape_hn``) and the
corpus build (``corpus_build``) are CLI-first: their public surface
is ``python -m src.data.<module>`` with explicit args. Materializing
the asset by shelling out to the CLI keeps a single source of truth
(``make ph-scrape`` and the Dagster asset do the same thing) and
makes the asset graph reproducible from the command line.
"""