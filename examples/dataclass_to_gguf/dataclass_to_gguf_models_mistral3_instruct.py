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
        name="mistral3-instruct",
        layers=[28] * 28,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=32000, languages=["en"], context_length=32768),
        attention=AttentionConfig(heads=32, kv_heads=8, rotary_dim=64, rope_base=1_000_000.0),
        projection=ProjectionConfig(input_dim=512, output_dim=256, activation=Activation.SILU, use_bias=False),
        head_norm=0.82,
    )

    proj = np.random.default_rng(10).standard_normal((256, 512)).astype(np.float16)

    path = write_example(
        "dataclass_to_gguf_models_mistral3_instruct",
        config,
        ModelArchitecture.MISTRAL3,
        tensors={"proj.weight": proj},
        description="Mistral3 instruct-style projection (float16).",
    )
    print(path)


if __name__ == "__main__":
    main()
