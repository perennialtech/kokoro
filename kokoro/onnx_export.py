from __future__ import annotations

from pathlib import Path
from typing import Union

import torch


def export_generator_onnx(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    shape_plan,
    model,
    output_path: Union[str, Path],
    opset_version: int,
) -> None:
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = list(shape_plan.input_order())
    output_names = ["audio"]

    dynamic_shapes = shape_plan.export_dynamic_shapes_trt_save(model)
    dynamic_axes = {
        "x": {2: "generator_frames"},
        "audio": {2: "audio_samples"},
    }
    for name in input_names:
        if name.startswith("source_"):
            dynamic_axes[name] = {2: f"{name}_frames"}

    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            example_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            opset_version=int(opset_version),
            dynamic_shapes=dynamic_shapes,
            dynamic_axes=dynamic_axes,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
