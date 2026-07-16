from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def build_engine_from_onnx(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    shapes: dict[str, dict[str, tuple[int, ...]]],
    input_order: tuple[str, ...],
    precision: str,
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
) -> None:
    import tensorrt as trt

    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")

    onnx_path = Path(onnx_path).resolve()
    engine_path = Path(engine_path).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Explicit batch is the default in TensorRT 10+.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()

    try:
        parsed = parser.parse_from_file(str(onnx_path))
    except TypeError:
        # Compatibility with older TensorRT Python bindings, whose API was:
        # parse_from_file(model_path, severity)
        parsed = parser.parse_from_file(str(onnx_path), trt.Logger.WARNING)

    if not parsed:
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parser failed for {onnx_path}:\n{errors}")

    for name in input_order:
        profile.set_shape(
            name,
            tuple(shapes["lower"][name]),
            tuple(shapes["preferred"][name]),
            tuple(shapes["upper"][name]),
        )

    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    if workspace_size is not None:
        workspace_size = int(workspace_size)
        if workspace_size <= 0:
            raise ValueError("workspace_size must be positive when provided")

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size,
        )

    if builder_optimization_level is not None:
        builder_optimization_level = int(builder_optimization_level)
        if not 0 <= builder_optimization_level <= 5:
            raise ValueError("builder_optimization_level must be between 0 and 5")

        config.builder_optimization_level = builder_optimization_level

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    with engine_path.open("wb") as f:
        f.write(bytes(serialized_engine))
