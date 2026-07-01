"""Dagster Definitions — Phase 3.1 (card t_7928b3e2).

This module is the entry point ``dagster dev`` loads via the
``PYTHONPATH=src dagster dev -m src.dagster.definitions`` CLI.

It bundles the five assets and the nightly schedule into a single
``Definitions`` object that Dagster's code-server can introspect.

Why a Definitions object (not @job / @repository)
-------------------------------------------------
``Definitions`` is the Dagster 1.6+ recommended pattern. The
older ``@repository`` decorator is deprecated and harder to test.
``Definitions`` is just a typed dict — Dagster builds the
in-memory lineage graph from it at process start.
"""

from __future__ import annotations

from dagster import Definitions, define_asset_job

from src.dagster_assets.assets import (
    company_embeddings,
    eval_benchmark,
    hn_show_posts,
    nightly_re_embedding_schedule,
    product_hunt_archive,
    yc_directory,
)


# Re-embedding job — materializes the whole corpus subgraph on the
# nightly schedule. The eval_benchmark asset is deliberately
# excluded from this job: it tracks eval-set freshness, not corpus
# state, and shouldn't need re-materialization on every corpus refresh.
nightly_re_embedding_job = define_asset_job(
    name="nightly_re_embedding_job",
    selection=[
        yc_directory.key,
        product_hunt_archive.key,
        hn_show_posts.key,
        company_embeddings.key,
    ],
    description=(
        "Phase 3.1 nightly re-embedding. Materializes the four "
        "corpus assets; company_embeddings is idempotent and "
        "skips the bge-m3 embed when no input snapshot has "
        "changed since the last successful build."
    ),
)


defs = Definitions(
    assets=[
        yc_directory,
        product_hunt_archive,
        hn_show_posts,
        company_embeddings,
        eval_benchmark,
    ],
    jobs=[nightly_re_embedding_job],
    schedules=[nightly_re_embedding_schedule],
)


__all__ = ["defs", "nightly_re_embedding_job", "nightly_re_embedding_schedule"]