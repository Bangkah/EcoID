"""Inference configuration, loaded from config/inference.toml (stdlib only)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "inference.toml"


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    threshold: float
    top_k: int = 3

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "InferenceConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        unknown = set(data) - {"model_name", "threshold", "top_k"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)
