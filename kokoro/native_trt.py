from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import torch

from .exceptions import (TensorRTDeserializationError, TensorRTExecutionError,
                         TensorRTShapeError)
from .telemetry import (NoOpProfileContext, ProfileContext, shape_attr,
                        tensor_nbytes)


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
            raise TensorRTDeserializationError(
                f"Failed to deserialize TensorRT plan: {engine_path}"
            )

        context = engine.create_execution_context()
        if context is None:
            raise TensorRTDeserializationError(
                "Failed to create TensorRT execution context"
            )

        self.runtime = runtime
        self.engine = engine
        self.context = context
        self.stream = torch.cuda.Stream()
        self._lock = threading.Lock()

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
            raise TensorRTDeserializationError("TensorRT engine has no inputs")
        if not self.output_names:
            raise TensorRTDeserializationError("TensorRT engine has no outputs")

    def run(
        self,
        inputs: dict[str, torch.Tensor],
        profile: Optional[ProfileContext] = None,
    ) -> dict[str, torch.Tensor]:
        profile = profile or NoOpProfileContext()

        with profile.span(
            "trt.validate_inputs",
            attrs={
                "input_names": ",".join(self.input_names),
                "output_names": ",".join(self.output_names),
            },
        ):
            missing = [name for name in self.input_names if name not in inputs]
            if missing:
                raise ValueError(f"Missing TensorRT input tensors: {missing}")

            for name in self.input_names:
                tensor = inputs[name]
                if not tensor.is_cuda:
                    raise ValueError(f"TensorRT input {name!r} must be a CUDA tensor")
                if not tensor.is_contiguous():
                    raise ValueError(f"TensorRT input {name!r} must be contiguous")

        with profile.span("trt.context_wait"):
            self._lock.acquire()

        try:
            with profile.span("trt.set_input_shapes") as span:
                for name in self.input_names:
                    tensor = inputs[name]
                    span.attrs[f"{name}.shape"] = shape_attr(tensor)
                    span.attrs[f"{name}.dtype"] = str(tensor.dtype)
                    span.attrs[f"{name}.bytes"] = tensor_nbytes(tensor)
                    if not self.context.set_input_shape(name, tuple(tensor.shape)):
                        raise TensorRTShapeError(
                            f"TensorRT rejected input shape for {name!r}: {tuple(tensor.shape)}"
                        )

            with profile.span("trt.infer_shapes") as span:
                infer_shapes = getattr(self.context, "infer_shapes", None)
                span.attrs["available"] = infer_shapes is not None
                if infer_shapes is not None:
                    unresolved = infer_shapes()
                    if unresolved:
                        span.attrs["unresolved"] = ",".join(str(x) for x in unresolved)
                        raise TensorRTShapeError(
                            "TensorRT could not infer all tensor shapes; unresolved tensors: "
                            f"{list(unresolved)}"
                        )

            outputs: dict[str, torch.Tensor] = {}
            with profile.span("trt.allocate_outputs") as span:
                total_output_bytes = 0
                for name in self.output_names:
                    shape = tuple(
                        int(dim) for dim in self.context.get_tensor_shape(name)
                    )
                    if any(dim < 0 for dim in shape):
                        raise TensorRTShapeError(
                            f"TensorRT output {name!r} has unresolved shape {shape}"
                        )

                    outputs[name] = torch.empty(
                        shape,
                        device="cuda",
                        dtype=self.tensor_dtypes[name],
                    )
                    nbytes = tensor_nbytes(outputs[name])
                    total_output_bytes += nbytes
                    span.attrs[f"{name}.shape"] = shape_attr(outputs[name])
                    span.attrs[f"{name}.dtype"] = str(outputs[name].dtype)
                    span.attrs[f"{name}.bytes"] = nbytes
                span.attrs["output_bytes"] = total_output_bytes
                profile.histogram("trt_output_bytes", float(total_output_bytes), {})

            with profile.span("trt.set_tensor_addresses"):
                for name in self.input_names:
                    self.context.set_tensor_address(name, int(inputs[name].data_ptr()))
                for name, tensor in outputs.items():
                    self.context.set_tensor_address(name, int(tensor.data_ptr()))

            current_stream = torch.cuda.current_stream()
            self.stream.wait_stream(current_stream)

            with torch.cuda.stream(self.stream):
                with profile.span(
                    "trt.execute_async_v3",
                    cuda=True,
                    attrs={"stream": int(self.stream.cuda_stream)},
                ):
                    if not self.context.execute_async_v3(
                        stream_handle=self.stream.cuda_stream
                    ):
                        raise TensorRTExecutionError("TensorRT execute_async_v3 failed")

            current_stream.wait_stream(self.stream)

            profile.counter(
                "trt_executions_total",
                1,
                (
                    {
                        "precision": profile._base_labels().get("precision", ""),
                        "status": "ok",
                    }
                    if hasattr(profile, "_base_labels")
                    else {"precision": "", "status": "ok"}
                ),
            )
            return outputs
        except Exception:
            profile.counter(
                "trt_executions_total",
                1,
                (
                    {
                        "precision": profile._base_labels().get("precision", ""),
                        "status": "error",
                    }
                    if hasattr(profile, "_base_labels")
                    else {"precision": "", "status": "error"}
                ),
            )
            raise
        finally:
            self._lock.release()
