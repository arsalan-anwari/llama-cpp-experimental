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
        name="projection-tensor-llama4",
        layers=[2, 4, 6, 8],
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=50000, languages=["en"], context_length=4096),
        attention=AttentionConfig(heads=12, kv_heads=4, rotary_dim=48, rope_base=5000.0),
        projection=ProjectionConfig(input_dim=16, output_dim=8, activation=Activation.SILU, use_bias=True),
        head_norm=1.1,
    )

    weights = np.linspace(-0.25, 0.25, num=128, dtype=np.float32).reshape(8, 16)
    bias = np.zeros((8,), dtype=np.float32)

    path = write_example(
        "dataclass_to_gguf_models_projection_tensor_llama4",
        config,
        ModelArchitecture.LLAMA4,
        tensors={"projection.weight": weights, "projection.bias": bias},
        description="LLAMA4 demo with projection weights (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
