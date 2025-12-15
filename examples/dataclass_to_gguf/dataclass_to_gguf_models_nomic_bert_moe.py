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
        name="nomic-bert-moe",
        layers=[8] * 8,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=40000, languages=["en", "fr"], context_length=2048),
        attention=AttentionConfig(heads=16, kv_heads=4, rotary_dim=32),
        head_norm=1.05,
        extras={"experts": 4},
    )

    gate = np.eye(4, dtype=np.int32)

    path = write_example(
        "dataclass_to_gguf_models_nomic_bert_moe",
        config,
        ModelArchitecture.NOMIC_BERT,
        tensors={"moe.router": gate},
        description="Nomic-BERT style mixture with int32 router table.",
    )
    print(path)


if __name__ == "__main__":
    main()
