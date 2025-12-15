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
        name="starcoder-code",
        layers=[24] * 24,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=49152, languages=["code"], context_length=8192),
        attention=AttentionConfig(heads=16, kv_heads=4, rotary_dim=64),
        head_norm=0.93,
        extras={"language_ids": ["python", "cpp", "rust"]},
    )

    code_embed = np.random.default_rng(9).standard_normal((256, 256)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_starcoder_code",
        config,
        ModelArchitecture.STARCODER,
        tensors={"code.embedding": code_embed},
        description="Starcoder code embeddings (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
