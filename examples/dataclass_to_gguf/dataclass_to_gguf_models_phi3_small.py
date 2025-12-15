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
        name="phi3-small",
        layers=[24] * 24,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=32064, languages=["en"], context_length=4096),
        attention=AttentionConfig(heads=16, kv_heads=4, rotary_dim=64),
        head_norm=1.0,
    )

    lm_head = np.random.default_rng(5).standard_normal((128, 256)).astype(np.float64)

    path = write_example(
        "dataclass_to_gguf_models_phi3_small",
        config,
        ModelArchitecture.PHI3,
        tensors={"lm_head.weight": lm_head},
        description="PHI3 demo with float64 output head.",
    )
    print(path)


if __name__ == "__main__":
    main()
