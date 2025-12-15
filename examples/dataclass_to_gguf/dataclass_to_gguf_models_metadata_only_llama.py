#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dataclass_to_gguf_models_base import (
    Activation,
    AdapterConfig,
    AttentionConfig,
    DemoModelConfig,
    Modality,
    ModelArchitecture,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="metadata-only-llama",
        layers=[4, 8, 12],
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=32000, languages=["en", "es"], context_length=8192),
        attention=AttentionConfig(heads=16, kv_heads=8, rotary_dim=64),
        projection=None,
        head_norm=0.8,
        adapter=AdapterConfig(alpha=32.0, beta=0.15, enabled=False, name="frozen"),
        extras={"notes": "pure metadata"},
    )

    path = write_example(
        "dataclass_to_gguf_models_metadata_only_llama",
        config,
        ModelArchitecture.LLAMA,
        tensors={},
        description="Metadata-only GGUF for LLAMA architecture demo.",
    )
    print(path)


if __name__ == "__main__":
    main()
