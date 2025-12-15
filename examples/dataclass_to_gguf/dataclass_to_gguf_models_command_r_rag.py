#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dataclass_to_gguf_models_base import (
    AdapterConfig,
    AttentionConfig,
    DemoModelConfig,
    Modality,
    ModelArchitecture,
    TrainingStats,
    write_example,
)


def main() -> None:
    config = DemoModelConfig(
        name="command-r-rag",
        layers=[30] * 30,
        modality=Modality.TEXT,
        stats=TrainingStats(vocab_size=64000, languages=["en"], context_length=8192),
        attention=AttentionConfig(heads=48, kv_heads=8, rotary_dim=128),
        head_norm=0.78,
        adapter=AdapterConfig(alpha=16.0, beta=0.05, enabled=True, name="rag"),
        extras={"retriever": "hybrid"},
    )

    rag_proj = np.random.default_rng(12).standard_normal((256, 256)).astype(np.float16)

    path = write_example(
        "dataclass_to_gguf_models_command_r_rag",
        config,
        ModelArchitecture.COMMAND_R,
        tensors={"rag.projection": rag_proj},
        description="Command-R style RAG projection (float16).",
    )
    print(path)


if __name__ == "__main__":
    main()
