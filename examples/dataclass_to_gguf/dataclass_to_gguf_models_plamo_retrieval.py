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
        name="plamo-retrieval",
        layers=[20] * 20,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=32000, languages=["ja", "en"], context_length=4096),
        attention=AttentionConfig(heads=20, kv_heads=4, rotary_dim=64),
        projection=ProjectionConfig(input_dim=256, output_dim=256, activation=Activation.RELU, use_bias=True),
        head_norm=0.9,
        extras={"retrieval_topk": 5},
    )

    retriever = np.random.default_rng(13).standard_normal((256, 256)).astype(np.float32)

    path = write_example(
        "dataclass_to_gguf_models_plamo_retrieval",
        config,
        ModelArchitecture.PLAMO,
        tensors={"retrieval.encoder": retriever},
        description="Plamo retrieval encoder matrix (float32).",
    )
    print(path)


if __name__ == "__main__":
    main()
