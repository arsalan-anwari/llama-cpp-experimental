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
        name="qwen3-moe",
        layers=[30] * 30,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=152064, languages=["zh", "en"], context_length=32768, pretraining_tokens=800_000_000_000),
        attention=AttentionConfig(heads=40, kv_heads=8, rotary_dim=128, rope_base=1000000.0),
        head_norm=0.6,
        extras={"experts": 8, "gating": "top2"},
    )

    gating = np.random.default_rng(4).standard_normal((8, 40)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_qwen3_moe",
        config,
        ModelArchitecture.QWEN3MOE,
        tensors={"moe.gate": gating},
        description="QWEN3 mixture with float32 gating matrix.",
    )
    print(path)


if __name__ == "__main__":
    main()
