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
        name="falcon-dense",
        layers=[60] * 60,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=65024, languages=["en"], context_length=2048),
        attention=AttentionConfig(heads=32, kv_heads=32, rotary_dim=64),
        head_norm=0.92,
    )

    attn_scale = np.full((32,), 0.7, dtype=np.int64)

    path = write_example(
        "dataclass_to_gguf_models_falcon_dense",
        config,
        ModelArchitecture.FALCON,
        tensors={"attention.scale": attn_scale},
        description="Falcon dense config with int64 scaling factors.",
    )
    print(path)


if __name__ == "__main__":
    main()
