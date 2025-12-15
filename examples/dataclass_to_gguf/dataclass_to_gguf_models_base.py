#!/usr/bin/env python3

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

# Ensure imports resolve to the repo root (avoids shadowing by local copies)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
sys.path = [str(_REPO_ROOT)] + [p for p in sys.path if p != str(_THIS_DIR)]

from convert_dataclass_to_gguf import ModelArchitecture, convert_dataclass_to_gguf


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class Activation(str, Enum):
    SILU = "silu"
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"


@dataclass
class TrainingStats:
    vocab_size: int
    languages: List[str] = field(default_factory=list)
    context_length: int = 4096
    pretraining_tokens: int | None = None
    dropout: float = 0.0


@dataclass
class AttentionConfig:
    heads: int
    kv_heads: int | None = None
    rotary_dim: int | None = None
    rope_base: float = 10000.0
    max_alibi_bias: float | None = None


class AdapterConfig(BaseModel):
    alpha: float = Field(..., gt=0)
    beta: float = 0.0
    enabled: bool = True
    name: str | None = None


@dataclass
class ProjectionConfig:
    input_dim: int
    output_dim: int
    activation: Activation = Activation.SILU
    use_bias: bool = True


@dataclass
class DemoModelConfig:
    name: str
    layers: List[int]
    modality: Modality
    stats: TrainingStats
    attention: AttentionConfig | None = None
    projection: ProjectionConfig | None = None
    head_norm: float = 1.0
    adapter: AdapterConfig | None = None
    extras: Dict[str, Any] = field(default_factory=dict)


def output_path(example_name: str) -> Path:
    base_dir = _REPO_ROOT / "models" / "custom"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{example_name}.gguf"


def write_example(
    example_name: str,
    payload: Any,
    architecture: ModelArchitecture,
    tensors: Dict[str, np.ndarray] | None = None,
    description: str | None = None,
    prefix: str | None = None,
) -> Path:
    return convert_dataclass_to_gguf(
        payload,
        output_path(example_name),
        architecture=architecture,
        metadata_prefix=prefix or example_name,
        tensor_overrides=tensors,
        description=description,
    )


__all__ = [
    "AdapterConfig",
    "Activation",
    "AttentionConfig",
    "DemoModelConfig",
    "Modality",
    "ModelArchitecture",
    "ProjectionConfig",
    "TrainingStats",
    "write_example",
]
