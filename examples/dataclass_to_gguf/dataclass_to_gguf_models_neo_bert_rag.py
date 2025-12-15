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
        name="neo-bert-rag",
        layers=[10] * 10,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=42000, languages=["en", "ja"], context_length=1536),
        attention=AttentionConfig(heads=18, kv_heads=6, rotary_dim=48, rope_base=1000.0),
        head_norm=1.2,
        extras={"retriever": "dense"},
    )

    codes = np.arange(1024, dtype=np.int8).reshape(16, 64)

    path = write_example(
        "dataclass_to_gguf_models_neo_bert_rag",
        config,
        ModelArchitecture.NEO_BERT,
        tensors={"retriever.codes": codes},
        description="NEO-BERT RAG codes stored as int8.",
    )
    print(path)


if __name__ == "__main__":
    main()
