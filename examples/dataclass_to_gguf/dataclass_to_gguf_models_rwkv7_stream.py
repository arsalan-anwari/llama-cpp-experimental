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
        name="rwkv7-stream",
        layers=[32] * 32,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=65536, languages=["en"], context_length=4096),
        attention=AttentionConfig(heads=8, kv_heads=8, rotary_dim=None),
        head_norm=1.0,
        extras={"state_groups": 8},
    )

    time_mix = np.random.default_rng(15).standard_normal((32, 8)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_rwkv7_stream",
        config,
        ModelArchitecture.RWKV7,
        tensors={"time_mix": time_mix},
        description="RWKV7 time mix parameters (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
