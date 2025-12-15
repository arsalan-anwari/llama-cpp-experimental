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
        name="bert-classifier",
        layers=[12] * 12,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=30522, languages=["en"], context_length=512),
        attention=AttentionConfig(heads=12, kv_heads=12, rotary_dim=None),
        head_norm=0.95,
    )

    classifier = np.arange(768, dtype=np.int8).reshape(3, 256)

    path = write_example(
        "dataclass_to_gguf_models_bert_classifier",
        config,
        ModelArchitecture.BERT,
        tensors={"classifier.weight": classifier},
        description="BERT classifier weights encoded as int8.",
    )
    print(path)


if __name__ == "__main__":
    main()
