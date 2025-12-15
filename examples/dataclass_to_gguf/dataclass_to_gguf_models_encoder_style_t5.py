#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dataclass_to_gguf_models_base import (
    Activation,
    AttentionConfig,
    DemoModelConfig,
    Modality,
    ModelArchitecture,
    ProjectionConfig,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="encoder-style-t5",
        layers=[6, 6, 6, 6],
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=32128, languages=["en"], context_length=1024),
        attention=AttentionConfig(heads=8, kv_heads=8, rotary_dim=None),
        projection=ProjectionConfig(input_dim=64, output_dim=64, activation=Activation.RELU, use_bias=False),
        head_norm=1.3,
    )

    enc = np.ones((4, 4, 4), dtype=np.float32)

    path = write_example(
        "dataclass_to_gguf_models_encoder_style_t5",
        config,
        ModelArchitecture.T5,
        tensors={"encoder.residual_scale": enc},
        description="T5 encoder-style config with 3D float32 tensor.",
    )
    print(path)


if __name__ == "__main__":
    main()
