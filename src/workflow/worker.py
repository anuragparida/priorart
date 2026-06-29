"""Temporal worker entrypoint — long-running process that hosts the activities.

Run with ``python -m src.workflow.worker`` (the Makefile wires this
into ``make worker`` / ``make dev``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.config import (
    TEMPORAL_ADDRESS,
    TEMPORAL_NAMESPACE,
    TEMPORAL_TASK_QUEUE,
)
from src.workflow.activities import (
    _reset_all_for_tests,
    ann_search,
    assemble_verdict,
    embed_idea,
    llm_compare_topk,
    market_scope_signal,
    web_fallback_if_empty,
)
from src.workflow.workflows import IdeaAnalysisWorkflow

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI args. All flags have sensible defaults from ``src.config``."""
    parser = argparse.ArgumentParser(
        prog="priorart-worker",
        description=(
            "Long-running Temporal worker for PriorArt. Polls the "
            "``priorart-idea-analysis`` task queue and executes "
            "``IdeaAnalysisWorkflow`` + its activities."
        ),
    )
    parser.add_argument(
        "--address",
        default=TEMPORAL_ADDRESS,
        help="Temporal server gRPC endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--namespace",
        default=TEMPORAL_NAMESPACE,
        help="Temporal namespace (default: %(default)s).",
    )
    parser.add_argument(
        "--task-queue",
        default=TEMPORAL_TASK_QUEUE,
        help="Task queue to poll (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Worker log level (default: %(default)s).",
    )
    parser.add_argument(
        "--reset-engine",
        action="store_true",
        help="Drop the cached SQLAlchemy engine before polling (test-only).",
    )
    return parser


async def _run(
    address: str,
    namespace: str,
    task_queue: str,
    reset_engine: bool = False,
) -> None:
    """Connect + register + poll. Returns only on Ctrl-C / SIGTERM."""
    if reset_engine:
        _reset_all_for_tests()

    logging.info(
        "worker: connecting to Temporal at %s (namespace=%s, task-queue=%s)",
        address,
        namespace,
        task_queue,
    )
    client = await Client.connect(
        address,
        namespace=namespace,
        data_converter=pydantic_data_converter,
    )

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[IdeaAnalysisWorkflow],
        activities=[
            embed_idea,
            ann_search,
            llm_compare_topk,
            market_scope_signal,
            assemble_verdict,
            web_fallback_if_empty,
        ],
    )

    logging.info("worker: ready; polling task queue %r", task_queue)
    await worker.run()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint — returns the process exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, loop.stop)

    try:
        loop.run_until_complete(
            _run(
                address=args.address,
                namespace=args.namespace,
                task_queue=args.task_queue,
                reset_engine=args.reset_engine,
            )
        )
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())