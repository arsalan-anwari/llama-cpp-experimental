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
    ProjectionConfig,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="qwen-chat",
        layers=[24] * 24,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=151936, languages=["zh", "en"], context_length=8192, pretraining_tokens=600_000_000_000),
        attention=AttentionConfig(heads=28, kv_heads=4, rotary_dim=80, rope_base=500000.0),
        projection=ProjectionConfig(input_dim=128, output_dim=64, activation=Activation.SWISH, use_bias=True),
        head_norm=0.7,
        adapter=AdapterConfig(alpha=8.0, beta=0.2, enabled=True, name="chat-adapter"),
    )

    down_proj = np.random.default_rng(2).standard_normal((64, 128)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_qwen_chat",
        config,
        ModelArchitecture.QWEN,
        tensors={"proj.down": down_proj},
        description="QWEN chat-style config with float32 down projection.",
    )
    print(path)


if __name__ == "__main__":
    main()
