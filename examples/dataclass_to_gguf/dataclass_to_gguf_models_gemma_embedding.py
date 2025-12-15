#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dataclass_to_gguf_models_base import (
    Activation,
    AttentionConfig,
    DemoModelConfig,
    Modality,
    ModelArchitecture,
    ProjectionConfig,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="gemma-embedding",
        layers=[18] * 18,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=256000, languages=["en"], context_length=8192),
        attention=AttentionConfig(heads=20, kv_heads=4, rotary_dim=64),
        projection=ProjectionConfig(input_dim=2048, output_dim=1024, activation=Activation.SILU, use_bias=False),
        head_norm=0.88,
    )

    embed = np.random.default_rng(7).standard_normal((512, 128)).astype(np.float16)

    path = write_example(
        "dataclass_to_gguf_models_gemma_embedding",
        config,
        ModelArchitecture.GEMMA_EMBEDDING,
        tensors={"token.embeddings": embed},
        description="Gemma embedding table encoded as float16.",
    )
    print(path)


if __name__ == "__main__":
    main()
