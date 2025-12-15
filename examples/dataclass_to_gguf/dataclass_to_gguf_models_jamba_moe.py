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
        name="jamba-moe",
        layers=[36] * 36,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=65536, languages=["en"], context_length=32768),
        attention=AttentionConfig(heads=64, kv_heads=8, rotary_dim=128),
        head_norm=0.75,
        extras={"experts": 16},
    )

    router = np.random.default_rng(8).standard_normal((16, 64)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_jamba_moe",
        config,
        ModelArchitecture.JAMBA,
        tensors={"router.scores": router},
        description="Jamba mixture router scores (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
