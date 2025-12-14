from __future__ import annotations

from dataclasses import dataclass, field, fields
import numpy as np
import gguf

from pydantic import BaseModel, field_validator
from pydantic import ConfigDict

# =============================================================================
# Pydantic validation layer (v2 compliant)
# =============================================================================

class TensorField(BaseModel):
    """
    Strict tensor wrapper to ensure GGUF compatibility.
    """

    value: np.ndarray

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @field_validator("value")
    @classmethod
    def validate_tensor(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("Tensor must be a numpy ndarray")

        # Enforce canonical authoring dtype
        if v.dtype != np.float32:
            raise TypeError(
                f"Unsupported tensor dtype {v.dtype}. "
                f"Only float32 is allowed for GGUF authoring."
            )

        if not v.flags["C_CONTIGUOUS"]:
            raise ValueError("Tensor must be C-contiguous")

        return v


class BitwiseExampleSchema(BaseModel):
    """
    User-facing schema that prevents invalid GGUF models.
    """

    # ---- metadata ----
    general_architecture: str
    demo_shiftA: int
    demo_shiftB: int

    # ---- tensors ----
    A: TensorField
    B: TensorField
    C: TensorField

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )


# =============================================================================
# GGUF model base (serialization layer)
# =============================================================================

@dataclass
class GGUFModel:
    """
    Dataclass-driven GGUF serializer.

    Field metadata:
      metadata={"gguf": "kv"}     -> GGUF metadata
      metadata={"gguf": "tensor"} -> GGUF tensor
    """

    def add_to_gguf(self, writer: gguf.GGUFWriter) -> None:
        # ---- metadata ----
        for f in fields(self):
            if f.metadata.get("gguf") == "kv":
                key = f.name.replace("_", ".")
                value = getattr(self, f.name)

                # general.architecture is set by GGUFWriter(arch=...)
                if key == "general.architecture":
                    continue

                if isinstance(value, str):
                    writer.add_string(key, value)
                elif isinstance(value, int):
                    writer.add_int32(key, value)
                else:
                    raise TypeError(f"Unsupported metadata type for {key}")

        # ---- tensors ----
        for f in fields(self):
            if f.metadata.get("gguf") == "tensor":
                tensor = getattr(self, f.name)

                if tensor.dtype != np.float32:
                    raise TypeError(
                        f"Tensor '{f.name}' must be float32, got {tensor.dtype}"
                    )

                writer.add_tensor(f.name, tensor)


# =============================================================================
# Concrete GGUF model
# =============================================================================

@dataclass
class BitwiseExample1(GGUFModel):
    # ---- metadata ----
    general_architecture: str = field(metadata={"gguf": "kv"})
    demo_shiftA: int          = field(metadata={"gguf": "kv"})
    demo_shiftB: int          = field(metadata={"gguf": "kv"})

    # ---- tensors ----
    A: np.ndarray = field(metadata={"gguf": "tensor"})
    B: np.ndarray = field(metadata={"gguf": "tensor"})
    C: np.ndarray = field(metadata={"gguf": "tensor"})


# =============================================================================
# Generator
# =============================================================================

def generate_bitwise_example_1(
    path: str = "models/custom/bitwise-nn/bitwise-example-1.gguf",
    n: int = 16,
) -> None:
    """
    Generate a minimal, tool-safe GGUF using F32 tensors.
    """

    # ---- user-facing validated schema ----
    schema = BitwiseExampleSchema(
        general_architecture="bitwise-nn",
        demo_shiftA=2,
        demo_shiftB=3,
        A=TensorField(value=np.random.randn(n, n).astype(np.float32)),
        B=TensorField(value=np.random.randn(n, n).astype(np.float32)),
        C=TensorField(value=np.zeros((n, n), dtype=np.float32)),
    )

    # ---- convert schema → GGUF dataclass ----
    model = BitwiseExample1(
        general_architecture=schema.general_architecture,
        demo_shiftA=schema.demo_shiftA,
        demo_shiftB=schema.demo_shiftB,
        A=schema.A.value,
        B=schema.B.value,
        C=schema.C.value,
    )

    writer = gguf.GGUFWriter(
        path,
        arch="bitwise-nn",
        use_temp_file=False,
    )

    # Minimal housekeeping key required by example tools
    writer.add_uint32("general.file_type", 0)

    model.add_to_gguf(writer)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("Wrote GGUF:", path)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    generate_bitwise_example_1()
