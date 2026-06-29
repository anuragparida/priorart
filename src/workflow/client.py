"""Temporal client helper for the FastAPI layer.

Phase 2.1: a singleton ``Client`` configured with the Pydantic v2
data converter (``temporalio.contrib.pydantic.pydantic_data_converter``).
That converter preserves Pydantic model types across the
activity/workflow boundary — without it, the SDK serialises Pydantic
models as plain dicts and the workflow code that does
``self._ann_result.hits[: input.top_k]`` fails with
``AttributeError: 'dict' object has no attribute 'hits'``.

The converter is a no-op for non-Pydantic payloads, so this is a
safe drop-in addition.
"""

from __future__ import annotations

import logging

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.config import TEMPORAL_ADDRESS, TEMPORAL_NAMESPACE

logger = logging.getLogger(__name__)


_client: Client | None = None


async def get_temporal_client() -> Client:
    """Return the singleton Temporal client (Pydantic-data-converter)."""
    global _client
    if _client is None:
        logger.info("temporal client: connecting to %s", TEMPORAL_ADDRESS)
        _client = await Client.connect(
            TEMPORAL_ADDRESS,
            namespace=TEMPORAL_NAMESPACE,
            data_converter=pydantic_data_converter,
        )
    return _client


def reset_client_for_tests() -> None:
    """Test hook — drop the cached client."""
    global _client
    _client = None


__all__ = ["get_temporal_client", "reset_client_for_tests"]