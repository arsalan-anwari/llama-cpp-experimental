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
        name="multitensor-precision-gptj",
        layers=[12, 12, 12],
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=64000, languages=["en", "de"], context_length=2048),
        attention=AttentionConfig(heads=10, kv_heads=10, rotary_dim=80),
        projection=ProjectionConfig(input_dim=32, output_dim=16, activation=Activation.GELU, use_bias=False),
        head_norm=0.9,
    )

    w16 = np.random.default_rng(0).standard_normal((16, 32)).astype(np.float16)
    w64 = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float64)

    path = write_example(
        "dataclass_to_gguf_models_multitensor_precision_gptj",
        config,
        ModelArchitecture.GPTJ,
        tensors={"proj.weight.f16": w16, "proj.weight.f64": w64},
        description="GPT-J style with both float16 and float64 tensors.",
    )
    print(path)


if __name__ == "__main__":
    main()
