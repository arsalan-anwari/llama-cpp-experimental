#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dataclass_to_gguf_models_base import (
    AttentionConfig,
    DemoModelConfig,
    Modality,
    ModelArchitecture,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="deepseek2-moe",
        layers=[30] * 30,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=200000, languages=["en", "zh"], context_length=32768),
        attention=AttentionConfig(heads=48, kv_heads=8, rotary_dim=128),
        head_norm=0.74,
        extras={"experts": 64},
    )

    load_balancing = np.random.default_rng(14).standard_normal((64,)).astype(np.float64)

    path = write_example(
        "dataclass_to_gguf_models_deepseek2_moe",
        config,
        ModelArchitecture.DEEPSEEK2,
        tensors={"moe.load_balancing": load_balancing},
        description="DeepSeek2 MOE load balancing as float64.",
    )
    print(path)


if __name__ == "__main__":
    main()
