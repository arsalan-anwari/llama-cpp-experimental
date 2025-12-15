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
        name="kv-heads-gptneox",
        layers=[24, 24],
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=52000, languages=["en"], context_length=4096, dropout=0.05),
        attention=AttentionConfig(heads=20, kv_heads=4, rotary_dim=64, max_alibi_bias=2.0),
        head_norm=1.0,
    )

    rope = np.arange(64, dtype=np.int16)

    path = write_example(
        "dataclass_to_gguf_models_kv_heads_gptneox",
        config,
        ModelArchitecture.GPTNEOX,
        tensors={"rope.freqs": rope},
        description="GPT-NeoX demo with int16 rotary table.",
    )
    print(path)


if __name__ == "__main__":
    main()
