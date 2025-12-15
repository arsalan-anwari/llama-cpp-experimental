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
        name="baichuan-multilingual",
        layers=[32] * 32,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=125696, languages=["zh", "en", "ru"], context_length=8192),
        attention=AttentionConfig(heads=32, kv_heads=8, rotary_dim=64),
        head_norm=0.85,
    )

    pos_embed = np.random.default_rng(6).standard_normal((256, 64)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_baichuan_multilingual",
        config,
        ModelArchitecture.BAICHUAN,
        tensors={"positional.embedding": pos_embed},
        description="Baichuan multilingual config with positional embeddings.",
    )
    print(path)


if __name__ == "__main__":
    main()
