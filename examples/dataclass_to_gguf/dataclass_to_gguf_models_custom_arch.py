#!/usr/bin/env python3
from __future__ import annotations

import sys
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
        name="custom-arch-generic",
        layers=[3, 5, 7],
        modality=Modality.MULTIMODAL,
        stats=TrainingStats(vocab_size=16000, languages=["en", "es"], context_length=1024, dropout=0.1),
        attention=AttentionConfig(heads=6, kv_heads=2, rotary_dim=24, rope_base=7500.0, max_alibi_bias=0.5),
        projection=ProjectionConfig(input_dim=24, output_dim=12, activation=Activation.RELU, use_bias=True),
        head_norm=1.0,
        extras={"custom_architecture": "dataclass-generic", "notes": "demonstrates mixed tensor dtypes"},
    )

    tensors = {
        "custom.embed": np.arange(48, dtype=np.float32).reshape(4, 12),
        "custom.router": np.arange(12, dtype=np.int16).reshape(3, 4),
        "custom.scale": np.linspace(0.1, 1.1, num=6, dtype=np.float64),
        # GGUF only supports signed integer tensor dtypes; use int8 for masks.
        "custom.mask": np.array([[1, 0], [0, 1]], dtype=np.int8),
    }

    path = write_example(
        "dataclass_to_gguf_custom_arch_dataclass_generic",
        config,
        ModelArchitecture.DATACLASS_GENERIC,
        tensors=tensors,
        description="Custom architecture tag with mixed tensor dtypes (f32, f64, i16, i8).",
        prefix="custom.generic",
    )
    print(path)


if __name__ == "__main__":
    main()
