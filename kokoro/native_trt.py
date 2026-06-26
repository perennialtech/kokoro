from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }

    if hasattr(trt.DataType, "UINT8"):
        mapping[trt.DataType.UINT8] = torch.uint8
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = torch.int64
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16

    try:
        return mapping[dtype]
    except KeyError as e:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}") from e


class NativeTRTEngine:
    def __init__(self, engine_path: Union[str, Path]):
        if not torch.cuda.is_available():
            raise RuntimeError("Native TensorRT execution requires CUDA")

        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as r:
            plan = r.read()

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT plan: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.runtime = runtime
        self.engine = engine
        self.context = context

        self.tensor_names: tuple[str, ...] = tuple(
            engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
        )
        self.tensor_modes = {
            name: engine.get_tensor_mode(name) for name in self.tensor_names
        }
        self.tensor_dtypes = {
            name: trt_dtype_to_torch(engine.get_tensor_dtype(name))
            for name in self.tensor_names
        }

        self.input_names: tuple[str, ...] = tuple(
            name
            for name in self.tensor_names
            if self.tensor_modes[name] == trt.TensorIOMode.INPUT
        )
        self.output_names: tuple[str, ...] = tuple(
            name
            for name in self.tensor_names
            if self.tensor_modes[name] == trt.TensorIOMode.OUTPUT
        )

        if not self.input_names:
            raise RuntimeError("TensorRT engine has no inputs")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no outputs")

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing TensorRT input tensors: {missing}")

        for name in self.input_names:
            tensor = inputs[name]
            if not tensor.is_cuda:
                raise ValueError(f"TensorRT input {name!r} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"TensorRT input {name!r} must be contiguous")

            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(
                    f"TensorRT rejected input shape for {name!r}: {tuple(tensor.shape)}"
                )

        infer_shapes = getattr(self.context, "infer_shapes", None)
        if infer_shapes is not None:
            unresolved = infer_shapes()
            if unresolved:
                raise RuntimeError(
                    "TensorRT could not infer all tensor shapes; unresolved tensors: "
                    f"{list(unresolved)}"
                )

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT output {name!r} has unresolved shape {shape}"
                )

            outputs[name] = torch.empty(
                shape,
                device="cuda",
                dtype=self.tensor_dtypes[name],
            )

        for name in self.input_names:
            self.context.set_tensor_address(name, int(inputs[name].data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return outputs
