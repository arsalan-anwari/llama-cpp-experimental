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
        name="mamba-sequence",
        layers=[24] * 24,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=65536, languages=["en"], context_length=16384),
        attention=AttentionConfig(heads=8, kv_heads=2, rotary_dim=32),
        head_norm=0.97,
    )

    ssm_kernel = np.linspace(0.0, 1.0, num=128, dtype=np.float32).reshape(16, 8)

    path = write_example(
        "dataclass_to_gguf_models_mamba_sequence",
        config,
        ModelArchitecture.MAMBA,
        tensors={"ssm.kernel": ssm_kernel},
        description="Mamba-style sequence kernel stored as float32.",
    )
    print(path)


if __name__ == "__main__":
    main()
