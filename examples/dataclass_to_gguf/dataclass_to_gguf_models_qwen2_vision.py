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
        name="qwen2-vision",
        layers=[12] * 12,
        modality=Modality.VISION,
        stats=TrainingStats(vocab_size=64000, languages=["en"], context_length=2048),
        attention=AttentionConfig(heads=16, kv_heads=8, rotary_dim=64),
        projection=ProjectionConfig(input_dim=256, output_dim=128, activation=Activation.SILU, use_bias=False),
        head_norm=1.0,
        extras={"image_size": 448},
    )

    vision_patch = np.random.default_rng(3).standard_normal((128, 256)).astype(np.float16)

    path = write_example(
        "dataclass_to_gguf_models_qwen2_vision",
        config,
        ModelArchitecture.QWEN2VL,
        tensors={"vision.patch_embed": vision_patch},
        description="QWEN2-VL style vision projector (float16).",
    )
    print(path)


if __name__ == "__main__":
    main()
