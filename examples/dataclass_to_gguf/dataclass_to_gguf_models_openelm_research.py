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
        name="openelm-research",
        layers=[16] * 16,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=50257, languages=["en"], context_length=2048),
        attention=AttentionConfig(heads=12, kv_heads=12, rotary_dim=32),
        head_norm=1.05,
        extras={"research": True},
    )

    lm = np.random.default_rng(11).standard_normal((256, 256)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_openelm_research",
        config,
        ModelArchitecture.OPENELM,
        tensors={"lm_head.weight": lm},
        description="OpenELM research head weights (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
