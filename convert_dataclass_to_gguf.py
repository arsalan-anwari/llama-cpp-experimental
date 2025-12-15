from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union, get_args, get_origin, List

import numpy as np
from pydantic import BaseModel, TypeAdapter

# Prefer the in-tree gguf package when available
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFWriter  # noqa: E402
from gguf.constants import GGUFEndian, GGUFValueType, MODEL_ARCH, MODEL_ARCH_NAMES  # noqa: E402

logger = logging.getLogger("convert-dataclass-to-gguf")


def _build_model_architecture_enum() -> type[Enum]:
    arch_entries: dict[str, str] = {arch.name: name for arch, name in MODEL_ARCH_NAMES.items()}
    arch_entries["DATACLASS_GENERIC"] = "dataclass-generic"
    # Add any new model architecture here....
    return Enum("ModelArchitecture", arch_entries, type=str)  # type: ignore[arg-type]


ModelArchitecture = _build_model_architecture_enum()


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
        if len(args) == 1:
            return _strip_optional(args[0])
    return annotation


def _unwrap_sequence(annotation: Any) -> tuple[Any | None, bool]:
    ann = _strip_optional(annotation)
    origin = get_origin(ann)
    if origin in (list, List, tuple, set, Sequence, Iterable):  # type: ignore[name-defined]
        args = get_args(ann)
        return (args[0] if args else None, True)
    return ann, False


def _normalize_key_piece(piece: Any) -> str:
    raw = str(piece)
    return raw.replace(" ", "_")


def _join_key(prefix: str | None, piece: str) -> str:
    if prefix:
        return f"{prefix}.{piece}"
    return piece


def _normalize_scalar(val: Any) -> Any:
    if isinstance(val, Enum):
        return _normalize_scalar(val.value)
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, Path):
        return str(val)
    return val


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_sequence(value: Any) -> bool:
    return (
        (isinstance(value, Sequence) or isinstance(value, set))
        and not isinstance(value, (str, bytes, bytearray, np.ndarray))
    )


_ALLOWED_TENSOR_DTYPES = {
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
}


def _validate_tensor_dtype(name: str, tensor: np.ndarray[Any, Any]) -> None:
    if tensor.dtype.type not in _ALLOWED_TENSOR_DTYPES:
        raise ValueError(
            f"Tensor {name!r} has unsupported dtype {tensor.dtype}; "
            "GGUFWriter currently allows only F16/F32/F64/I8/I16/I32/I64."
        )


def _guess_int_value_type(value: int, annotation: Any | None) -> GGUFValueType:
    ann = _strip_optional(annotation)
    unsigned = ann in (np.uint8, np.uint16, np.uint32, np.uint64)

    if unsigned and value < 0:
        raise ValueError("Negative value provided for unsigned integer field")

    if ann in (np.int8, np.uint8):
        return GGUFValueType.UINT8 if unsigned else GGUFValueType.INT8
    if ann in (np.int16, np.uint16):
        return GGUFValueType.UINT16 if unsigned else GGUFValueType.INT16
    if ann in (np.int32, np.uint32):
        return GGUFValueType.UINT32 if unsigned else GGUFValueType.INT32
    if ann in (np.int64, np.uint64):
        return GGUFValueType.UINT64 if unsigned else GGUFValueType.INT64

    if unsigned:
        if value <= 0xFF:
            return GGUFValueType.UINT8
        if value <= 0xFFFF:
            return GGUFValueType.UINT16
        if value <= 0xFFFFFFFF:
            return GGUFValueType.UINT32
        return GGUFValueType.UINT64

    if -0x80 <= value <= 0x7F:
        return GGUFValueType.INT8
    if -0x8000 <= value <= 0x7FFF:
        return GGUFValueType.INT16
    if -(2**31) <= value <= (2**31 - 1):
        return GGUFValueType.INT32
    return GGUFValueType.INT64


def _guess_scalar_type(value: Any, annotation: Any | None) -> GGUFValueType:
    val = _normalize_scalar(value)
    ann = _strip_optional(annotation)

    if isinstance(val, bool) or ann is bool:
        return GGUFValueType.BOOL
    if isinstance(val, (float, np.floating)) or ann in (float, np.float16, np.float32, np.float64):
        if ann in (np.float64,) or isinstance(val, np.float64):
            return GGUFValueType.FLOAT64
        return GGUFValueType.FLOAT32
    if isinstance(val, (str, bytes, bytearray)) or ann is str:
        return GGUFValueType.STRING
    if isinstance(val, (int, np.integer)) or ann in (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        return _guess_int_value_type(int(val), ann)

    raise TypeError(f"Unsupported scalar type for value {value!r} with annotation {annotation!r}")


def _array_sub_type(values: Sequence[Any], annotation: Any | None) -> GGUFValueType:
    if not values:
        raise ValueError("Cannot encode empty sequences into GGUF arrays")

    elem_ann, is_seq = _unwrap_sequence(annotation)
    if is_seq:
        annotation = elem_ann

    sub_types = {_guess_scalar_type(_normalize_scalar(v), annotation) for v in values}
    if len(sub_types) != 1:
        raise TypeError("Mixed element types are not supported inside GGUF arrays")
    return sub_types.pop()


def _import_object(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError("Target must be in the form module:object")
    module = importlib.import_module(module_name)
    obj: Any = module
    for piece in attr.split("."):
        obj = getattr(obj, piece)
    return obj


def _ensure_instance(obj: Any, payload: Mapping[str, Any] | None) -> Any:
    if inspect.isclass(obj):
        if issubclass(obj, BaseModel):
            return obj.model_validate(payload or {})
        if is_dataclass(obj):
            adapter = TypeAdapter(obj)
            try:
                return adapter.validate_python(payload or {})
            except Exception:
                return obj(**(payload or {}))
        raise TypeError("Provided target is not a dataclass or pydantic BaseModel")
    if callable(obj):
        return obj(payload or {})
    return obj


class MetadataEncoder:
    def __init__(
        self,
        writer: GGUFWriter,
        *,
        prefix: str | None,
        type_overrides: Mapping[str, GGUFValueType | tuple[GGUFValueType, GGUFValueType]] | None = None,
        tensor_names: set[str] | None = None,
    ):
        self.writer = writer
        self.prefix = prefix
        self.type_overrides = type_overrides or {}
        self.tensor_names: set[str] = tensor_names or set()

    def _apply_override(
        self, key: str, default_type: GGUFValueType, default_sub_type: GGUFValueType | None
    ) -> tuple[GGUFValueType, GGUFValueType | None]:
        override = self.type_overrides.get(key)
        if override is None:
            return default_type, default_sub_type
        if isinstance(override, tuple):
            return override
        return override, default_sub_type

    def encode(self, value: Any, *, annotation: Any | None = None, key_prefix: str | None = None) -> None:
        if value is None:
            return

        if isinstance(value, Enum):
            self.encode(value.value, annotation=annotation, key_prefix=key_prefix)
            return

        if isinstance(value, BaseModel):
            hints = {name: field.annotation for name, field in value.model_fields.items()}
            for name, item in value.model_dump(mode="python").items():
                self.encode(item, annotation=hints.get(name), key_prefix=_join_key(key_prefix, _normalize_key_piece(name)))
            return

        if is_dataclass(value):
            hints = {f.name: f.type for f in fields(value)}
            for field_def in fields(value):
                self.encode(
                    getattr(value, field_def.name),
                    annotation=hints.get(field_def.name),
                    key_prefix=_join_key(key_prefix, _normalize_key_piece(field_def.name)),
                )
            return

        if isinstance(value, np.ndarray):
            tensor_key = key_prefix or (self.prefix or "tensor")
            _validate_tensor_dtype(tensor_key, value)
            if tensor_key in self.tensor_names:
                raise ValueError(f"Duplicate tensor name {tensor_key!r}")
            self.tensor_names.add(tensor_key)
            self.writer.add_tensor(tensor_key, value)
            shape_key = f"{tensor_key}.shape"
            dtype_key = f"{tensor_key}.dtype"
            self.writer.add_key_value(shape_key, list(value.shape), GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
            self.writer.add_string(dtype_key, str(value.dtype))
            return

        if _is_mapping(value):
            for map_key, map_val in value.items():
                part = _normalize_key_piece(map_key)
                self.encode(map_val, key_prefix=_join_key(key_prefix, part))
            return

        if _is_sequence(value):
            if len(value) == 0:
                return

            elem_hint, is_container = _unwrap_sequence(annotation)
            container = list(value)

            if any(_is_mapping(v) or is_dataclass(v) or isinstance(v, BaseModel) for v in container):
                for idx, item in enumerate(container):
                    self.encode(item, annotation=elem_hint, key_prefix=_join_key(key_prefix, str(idx)))
                return

            normalized = [_normalize_scalar(v) for v in container]
            subtype = _array_sub_type(normalized, elem_hint)
            key = key_prefix or self.prefix or "array"
            vtype, subtype = self._apply_override(key, GGUFValueType.ARRAY, subtype)
            self.writer.add_key_value(key, normalized, vtype, sub_type=subtype)
            return

        scalar = _normalize_scalar(value)
        key = key_prefix or self.prefix or "field"
        vtype, sub_type = self._apply_override(key, _guess_scalar_type(scalar, annotation), None)
        self.writer.add_key_value(key, scalar, vtype, sub_type=sub_type)


def convert_dataclass_to_gguf(
    data: Any,
    output_path: str | os.PathLike[str],
    *,
    architecture: ModelArchitecture | str = ModelArchitecture.DATACLASS_GENERIC,
    metadata_prefix: str | None = None,
    name: str | None = None,
    description: str | None = None,
    tensor_overrides: Mapping[str, np.ndarray] | None = None,
    type_overrides: Mapping[str, GGUFValueType | tuple[GGUFValueType, GGUFValueType]] | None = None,
    endianess: GGUFEndian = GGUFEndian.LITTLE,
    use_temp_file: bool = False,
) -> Path:
    arch = architecture
    if isinstance(arch, str):
        normalized = arch.upper().replace("-", "_")
        arch = ModelArchitecture.__members__.get(normalized) or next(
            (member for member in ModelArchitecture if member.value == arch), None
        )
        if arch is None:
            raise ValueError(f"Unknown architecture {architecture!r}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(output, arch.value, use_temp_file=use_temp_file, endianess=endianess)

    model_name = name or getattr(data, "__class__", type("anon", (), {})).__name__
    writer.add_name(model_name)
    writer.add_type(model_name)
    if description:
        writer.add_description(description)

    normalized_prefix = _normalize_key_piece(metadata_prefix) if metadata_prefix else _normalize_key_piece(model_name.lower())

    encoder = MetadataEncoder(writer, prefix=normalized_prefix, type_overrides=type_overrides)
    encoder.encode(data, key_prefix=normalized_prefix)

    for tensor_name, tensor_value in (tensor_overrides or {}).items():
        _validate_tensor_dtype(tensor_name, tensor_value)
        if tensor_name in encoder.tensor_names:
            raise ValueError(f"Tensor {tensor_name!r} already added from dataclass contents")
        encoder.tensor_names.add(tensor_name)
        writer.add_tensor(tensor_name, tensor_value)
        writer.add_key_value(f"{tensor_name}.shape", list(tensor_value.shape), GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
        writer.add_string(f"{tensor_name}.dtype", str(tensor_value.dtype))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info("Wrote GGUF file to %s", output)
    return output


def _parse_tensor_arg(arg: str) -> tuple[str, np.ndarray]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Tensor arguments must look like name=path.npy")
    name, path = arg.split("=", 1)
    tensor = np.load(Path(path), allow_pickle=False)
    return name, tensor


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serialize dataclass or pydantic models into GGUF.")
    parser.add_argument("--target", required=True, help="Python path to the dataclass or pydantic model (module:object).")
    parser.add_argument("--data", help="Path to JSON payload used to instantiate the model.")
    parser.add_argument("--output", required=True, help="Where to write the GGUF file.")
    parser.add_argument(
        "--arch",
        choices=[m.name.lower() for m in ModelArchitecture],
        default=ModelArchitecture.DATACLASS_GENERIC.name.lower(),
        help="Model architecture tag to embed into GGUF metadata.",
    )
    parser.add_argument("--metadata-prefix", help="Optional prefix for generated metadata keys.")
    parser.add_argument("--name", help="Overrides general.name metadata.")
    parser.add_argument("--description", help="Adds general.description metadata.")
    parser.add_argument(
        "--tensor",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional .npy tensor to include. May be provided multiple times.",
    )
    parser.add_argument("--big-endian", action="store_true", help="Emit GGUF data using big endian encoding.")
    parser.add_argument("--use-temp-file", action="store_true", help="Buffer tensors to disk before writing.")
    return parser.parse_args(argv)


def _load_payload(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    target_obj = _import_object(args.target)
    payload = _load_payload(args.data)
    instance = _ensure_instance(target_obj, payload)

    tensor_overrides = dict(_parse_tensor_arg(t) for t in args.tensor)
    architecture = args.arch.upper()

    convert_dataclass_to_gguf(
        instance,
        args.output,
        architecture=architecture,
        metadata_prefix=args.metadata_prefix,
        name=args.name,
        description=args.description,
        tensor_overrides=tensor_overrides,
        endianess=GGUFEndian.BIG if args.big_endian else GGUFEndian.LITTLE,
        use_temp_file=bool(args.use_temp_file),
    )


if __name__ == "__main__":
    main()
