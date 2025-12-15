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
        name="arcee-agent",
        layers=[14] * 14,
        modality=Modality.MULTIMODAL,
        stats=TrainingStats(vocab_size=48000, languages=["en"], context_length=2048),
        attention=AttentionConfig(heads=16, kv_heads=4, rotary_dim=32),
        head_norm=1.0,
        extras={"agent_tools": ["search", "code"]},
    )

    tool_bias = np.arange(16, dtype=np.int32)

    path = write_example(
        "dataclass_to_gguf_models_arcee_agent",
        config,
        ModelArchitecture.ARCEE,
        tensors={"tool.bias": tool_bias},
        description="Arcee agent biases stored as int32.",
    )
    print(path)


if __name__ == "__main__":
    main()
