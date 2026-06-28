"""Configuration loader for the eval harness.

A retrieval config is a tiny YAML file with the fields the runner
needs to issue a query and write a leaderboard row. We deliberately
do NOT model every knob in the system here — Phase 1 only has one
config (dense_bge_m3). Phase 2 will add bm25 / hybrid / rerank as
siblings of this file.

Why a typed loader and not raw ``yaml.safe_load`` at the call site
- One place to validate the file shape (typer exits with a clear
  message if a required key is missing, instead of a cryptic
  KeyError halfway through the run).
- Easy to extend in Phase 2 (add bm25 / hybrid configs by adding
  fields here, runner just keeps working).
- Easier to test (unit tests construct the dataclass directly
  instead of building YAML strings).

Schema
------
::

    name: str             # short slug written into leaderboard.csv
    embedding_model: str  # for the leaderboard's ``embedding_model`` column
    embedding_dim: int
    top_k: int            # how many hits to fetch per query
    api_url: str          # POST /search target
    notes: str            # free text, written into the ``notes`` column

Anything extra in the YAML is silently dropped — we don't want a
typo'd ``to_k:`` to silently truncate fetches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_REQUIRED = ("name", "embedding_model", "embedding_dim", "top_k", "api_url")
_KNOWN = _REQUIRED + ("notes",)


@dataclass(frozen=True)
class RetrievalConfig:
    """A single retrieval configuration."""

    name: str
    embedding_model: str
    embedding_dim: int
    top_k: int
    api_url: str
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "RetrievalConfig":
        """Load + validate a YAML config file.

        Raises ``ValueError`` with a precise message if a required
        field is missing or the wrong type. Unknown keys are ignored
        (not an error — keeps forward-compat).
        """
        with open(path) as f:
            raw: Any = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")

        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise ValueError(f"{path}: missing required keys: {missing}")

        # Type checks (loud, not silent)
        if not isinstance(raw["embedding_dim"], int):
            raise ValueError(f"{path}: embedding_dim must be int, got {type(raw['embedding_dim'])}")
        if not isinstance(raw["top_k"], int):
            raise ValueError(f"{path}: top_k must be int, got {type(raw['top_k'])}")
        if not isinstance(raw["name"], str) or not raw["name"]:
            raise ValueError(f"{path}: name must be non-empty string")
        if not isinstance(raw["embedding_model"], str) or not raw["embedding_model"]:
            raise ValueError(f"{path}: embedding_model must be non-empty string")
        if not isinstance(raw["api_url"], str) or not raw["api_url"]:
            raise ValueError(f"{path}: api_url must be non-empty string")

        return cls(
            name=raw["name"],
            embedding_model=raw["embedding_model"],
            embedding_dim=raw["embedding_dim"],
            top_k=raw["top_k"],
            api_url=raw["api_url"],
            notes=str(raw.get("notes", "")),
        )