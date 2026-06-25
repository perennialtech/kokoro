"""Offline Torch-TensorRT AOT compiler for Kokoro's ISTFTNet generator.

This compiles KokoroGenerateWithSourcePyramid. It intentionally does not compile
text duration, ProsodyPredictor.F0Ntrain, Decoder.decode_features, harmonic
feature generation, or harmonic/source pyramid generation.

Keeping Decoder.decode_features outside TensorRT avoids a fragile dynamic-shape
proof through the decoder upsample path that can make TensorRT reject the kMIN
profile with elementwise-add mismatches such as 20 != 40.
"""

import argparse
import hashlib
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from .config import save_trt_metadata
from .model import KModel, KokoroGenerateWithSourcePyramid
from .trt import (TRT_ENGINE_FILENAME, TensorRTDynamicShapeProfile,
                  generator_frame_count, generator_profile_shapes)


def sha256_file(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    p = Path(path)
    if not p.exists() or not p.is_file():
        return None

    h = hashlib.sha256()
    with open(p, "rb") as r:
        for chunk in iter(lambda: r.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:
        return None


def get_torch_tensorrt_version(torch_tensorrt: Any) -> Optional[str]:
    return str(getattr(torch_tensorrt, "__version__", None))


def metadata_for_compile(
    kmodel: KModel,
    profile: TensorRTDynamicShapeProfile,
    precision: str,
    shapes: dict[str, dict[str, tuple[int, ...]]],
    torch_tensorrt: Any,
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
) -> dict[str, Any]:
    major, minor = torch.cuda.get_device_capability()
    return {
        "artifact_type": "kokoro_generator_with_source_pyramid_tensorrt",
        "format_version": 1,
        "engine_file": TRT_ENGINE_FILENAME,
        "repo_id": kmodel.repo_id,
        "checkpoint": {
            "path": getattr(kmodel, "model_path", None),
            "sha256": sha256_file(getattr(kmodel, "model_path", None)),
        },
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": f"sm_{major}{minor}",
        },
        "versions": {
            "torch": torch.__version__,
            "tensorrt": get_tensorrt_version(),
            "torch_tensorrt": get_torch_tensorrt_version(torch_tensorrt),
        },
        "precision": precision,
        "workspace_size": workspace_size,
        "builder_optimization_level": builder_optimization_level,
        "profile": asdict(profile),
        "shapes": {
            group: {name: list(shape) for name, shape in specs.items()}
            for group, specs in shapes.items()
        },
    }


TRT_INPUT_BASE_ORDER = ("x", "ref_s")


def trt_input_order(kmodel: KModel) -> tuple[str, ...]:
    return (
        *TRT_INPUT_BASE_ORDER,
        *(f"source_{i}" for i in range(len(kmodel.decoder.source_channels()))),
    )


def tensorrt_inputs_from_profile(
    shapes: dict[str, dict[str, tuple[int, ...]]],
    dtype: torch.dtype,
    torch_tensorrt: Any,
    input_order: tuple[str, ...],
) -> list[Any]:
    def shape(group: str, name: str) -> tuple[int, ...]:
        return tuple(int(dim) for dim in shapes[group][name])

    # Do not set Input.name here. With a vararg source pyramid, exported input
    # names are Torch-generated implementation details. Positional profiles are
    # more robust and exactly match example_inputs/dynamic_shapes order.
    return [
        torch_tensorrt.Input(
            min_shape=shape("min", name),
            opt_shape=shape("opt", name),
            max_shape=shape("max", name),
            dtype=dtype,
        )
        for name in input_order
    ]


def example_tensors_from_profile(
    shapes: dict[str, dict[str, tuple[int, ...]]],
    dtype: torch.dtype,
    input_order: tuple[str, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(
            tuple(int(dim) for dim in shapes["opt"][name]),
            device="cuda",
            dtype=dtype,
        )
        for name in input_order
    )


def torch_tensorrt_compile_supports_kwarg(torch_tensorrt: Any, name: str) -> bool:
    """
    Return whether the public Torch-TensorRT compiler appears to accept a kwarg.

    Important: Kokoro intentionally uses torch_tensorrt.compile(..., ir="dynamo")
    rather than torch_tensorrt.dynamo.compile directly. In several
    Torch-TensorRT releases the lower-level dynamo compiler accepts arbitrary
    **kwargs but does not honor output_format='exported_program', returning a
    torch.fx.GraphModule instead. The top-level compiler is the API that owns
    output-format conversion.
    """
    try:
        signature = inspect.signature(torch_tensorrt.compile)
    except (TypeError, ValueError):
        return True

    if name in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def compile_exported_program_with_tensorrt(
    torch_tensorrt: Any,
    exported_program: torch.export.ExportedProgram,
    inputs: list[Any],
    enabled_precisions: set[torch.dtype],
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
) -> Any:
    """
    Compile an already-exported dynamic Kokoro graph.
    """
    kwargs: dict[str, Any] = {
        "ir": "dynamo",
        "inputs": inputs,
        "enabled_precisions": enabled_precisions,
        "require_full_compilation": True,
    }

    if workspace_size is not None:
        workspace_size = int(workspace_size)
        if workspace_size <= 0:
            raise ValueError("workspace_size must be positive when provided")

        if torch_tensorrt_compile_supports_kwarg(torch_tensorrt, "workspace_size"):
            kwargs["workspace_size"] = workspace_size

    if builder_optimization_level is not None:
        builder_optimization_level = int(builder_optimization_level)
        if not 0 <= builder_optimization_level <= 5:
            raise ValueError("builder_optimization_level must be between 0 and 5")

        if torch_tensorrt_compile_supports_kwarg(
            torch_tensorrt,
            "optimization_level",
        ):
            kwargs["optimization_level"] = builder_optimization_level
        elif torch_tensorrt_compile_supports_kwarg(
            torch_tensorrt,
            "builder_optimization_level",
        ):
            kwargs["builder_optimization_level"] = builder_optimization_level

    compiled = torch_tensorrt.compile(exported_program, **kwargs)

    return compiled


def affine_relation_for_points(
    name: str,
    fn,
    points: tuple[int, ...],
) -> tuple[int, int]:
    """
    Return slope/intercept for a frame-count helper:

        fn(frames) == slope * frames + intercept

    and verify that the relationship is linear across representative points.
    """
    y1 = int(fn(1))
    y2 = int(fn(2))
    slope = y2 - y1
    intercept = y1 - slope

    if slope <= 0:
        raise ValueError(f"{name} must have positive slope, got {slope}")

    check_points = {1, 2}
    check_points.update(int(point) for point in points)

    for frames in check_points:
        actual = int(fn(frames))
        expected = slope * frames + intercept
        if actual != expected:
            raise ValueError(
                f"{name} is not affine in frame count: "
                f"{name}({frames})={actual}, expected {expected} from "
                f"{slope} * frames + {intercept}"
            )

    return slope, intercept


def affine_relation_for_profile(
    name: str,
    fn,
    profile: TensorRTDynamicShapeProfile,
) -> tuple[int, int]:
    return affine_relation_for_points(
        name,
        fn,
        (
            int(profile.min_frames),
            int(profile.opt_frames),
            int(profile.max_frames),
        ),
    )


def dim_expr(base, slope: int, intercept: int):
    expr = base if slope == 1 else slope * base
    if intercept:
        expr = expr + intercept
    return expr


def torch_export_dynamic_shapes(
    kmodel: KModel,
    profile: TensorRTDynamicShapeProfile,
):
    """
    torch_tensorrt.Input can describe min/opt/max ranges, but it cannot express
    that several dynamic input dimensions are affine functions of the same
    symbolic generator-frame dimension.

    The generator-only export needs only this relationship:

        x.shape[2]        = generator_frames
        source_i.shape[2] = source_i_frame_count(generator_frames)

    Decoder.decode_features is intentionally outside TensorRT, so TensorRT no
    longer has to prove the synthesis_frames -> generator_frames relationship
    through the decoder upsample path.
    """
    from torch.export import Dim

    min_generator_frames = generator_frame_count(kmodel, profile.min_frames)
    opt_generator_frames = generator_frame_count(kmodel, profile.opt_frames)
    max_generator_frames = generator_frame_count(kmodel, profile.max_frames)

    if min_generator_frames < 3:
        required_min = profile.min_frames
        while generator_frame_count(kmodel, required_min) < 3:
            required_min += 1

        raise ValueError(
            "TensorRT export requires generator input length >= 3. "
            f"With min_frames={profile.min_frames}, generator-frame length is "
            f"{min_generator_frames}. Use --min-frames {required_min} or higher."
        )

    generator_frames = Dim(
        "generator_frames",
        min=int(min_generator_frames),
        max=int(max_generator_frames),
    )

    # Important:
    #
    # KokoroGenerateWithSourcePyramid.forward has this signature:
    #
    #     forward(x, ref_s, *source_pyramid)
    #
    # torch.export binds the varargs as one formal parameter. Therefore the input
    # pytree seen by torch.export is:
    #
    #     (x, ref_s, (source_0, source_1, ...))
    #
    # dynamic_shapes must match that bound pytree structure. Torch-TensorRT
    # profile inputs remain flat elsewhere because they describe the flattened
    # ExportedProgram graph inputs.
    base_dynamic_shapes: tuple[dict[int, Any], ...] = (
        {2: generator_frames},  # x: [1, C, generator_frames]
        {},  # ref_s: [1, 256]
    )

    profile_generator_points = (
        int(min_generator_frames),
        int(opt_generator_frames),
        int(max_generator_frames),
    )

    source_dynamic_shapes: list[dict[int, Any]] = []
    for i in range(len(kmodel.decoder.source_channels())):
        source_slope, source_intercept = affine_relation_for_points(
            f"source_{i}_frame_count_from_generator_frames",
            lambda generator_frame_count_value, i=i: (
                kmodel.decoder.generator.source_frame_lengths(
                    generator_frame_count_value
                )[i]
            ),
            profile_generator_points,
        )
        source_dynamic_shapes.append(
            {
                2: dim_expr(
                    generator_frames,
                    source_slope,
                    source_intercept,
                )
            }
        )

    return (*base_dynamic_shapes, tuple(source_dynamic_shapes))


def validate_decoder_generate_profile_shapes(
    module: torch.nn.Module,
    shapes: dict[str, dict[str, tuple[int, ...]]],
    dtype: torch.dtype,
    input_order: tuple[str, ...],
    groups: tuple[str, ...] = ("min",),
) -> None:
    """
    Run the exact tensors that will be used for TensorRT profile points through
    the PyTorch module before invoking TensorRT.

    This is intentionally cheap by default: validating only kMIN catches the
    "Profile kMIN values are not self-consistent" class of issue without running
    a large max-frame render.
    """

    with torch.inference_mode():
        for group in groups:
            example_inputs = tuple(
                torch.zeros(
                    tuple(int(dim) for dim in shapes[group][name]),
                    device="cuda",
                    dtype=dtype,
                )
                for name in input_order
            )

            try:
                output = module(*example_inputs)
            except Exception as e:
                formatted_shapes = {
                    name: tuple(int(dim) for dim in shapes[group][name])
                    for name in input_order
                }
                raise RuntimeError(
                    f"Decoder/generator profile point {group!r} is not "
                    f"self-consistent in PyTorch. Shapes: {formatted_shapes}"
                ) from e

            if output.dim() != 3 or output.shape[0] != 1 or output.shape[1] != 1:
                raise RuntimeError(
                    f"Decoder/generator profile point {group!r} returned an "
                    f"unexpected output shape: {tuple(output.shape)}"
                )

            del output
            del example_inputs

    torch.cuda.empty_cache()


def compile_generator_with_source_pyramid(
    kmodel: KModel,
    output_dir: Path,
    profile: TensorRTDynamicShapeProfile,
    precision: str,
    workspace_size: Optional[int] = 512 * 1024 * 1024,
    builder_optimization_level: Optional[int] = 0,
    validate_profile: bool = True,
) -> None:
    try:
        import torch_tensorrt
    except ImportError as e:
        raise ImportError(
            "TensorRT compilation requires torch-tensorrt. Install Torch-TensorRT "
            "matching your PyTorch/CUDA/TensorRT stack."
        ) from e

    profile.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Torch-TensorRT compilation requires CUDA")

    dtype = torch.float16 if precision == "fp16" else torch.float32

    kmodel.prepare_for_tensorrt_export()
    kmodel.to(device="cuda", dtype=dtype)

    module = (
        KokoroGenerateWithSourcePyramid(kmodel)
        .eval()
        .to(
            device="cuda",
            dtype=dtype,
        )
    )

    shapes = generator_profile_shapes(kmodel, profile)
    input_order = trt_input_order(kmodel)

    if validate_profile:
        validate_decoder_generate_profile_shapes(
            module=module,
            shapes=shapes,
            dtype=dtype,
            input_order=input_order,
            groups=("min",),
        )

    inputs = tensorrt_inputs_from_profile(
        shapes=shapes,
        dtype=dtype,
        torch_tensorrt=torch_tensorrt,
        input_order=input_order,
    )
    example_inputs = example_tensors_from_profile(
        shapes=shapes,
        dtype=dtype,
        input_order=input_order,
    )
    dynamic_shapes = torch_export_dynamic_shapes(
        kmodel=kmodel,
        profile=profile,
    )

    enabled_precisions = {torch.float16} if precision == "fp16" else {torch.float32}

    with torch.inference_mode():
        exported_program = torch.export.export(
            module,
            example_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

        torch.cuda.empty_cache()

        compiled = compile_exported_program_with_tensorrt(
            torch_tensorrt=torch_tensorrt,
            exported_program=exported_program,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_size,
            builder_optimization_level=builder_optimization_level,
        )

    engine_path = output_dir / TRT_ENGINE_FILENAME

    # torch_tensorrt.save calls torch.export on the compiled wrapper, which is
    # a flat GraphModule taking example_inputs directly. Flatten dynamic_shapes
    # to match the structure of example_inputs.
    flat_dynamic_shapes = tuple(dynamic_shapes[:2]) + tuple(dynamic_shapes[2])

    torch_tensorrt.save(
        compiled,
        str(engine_path),
        arg_inputs=example_inputs,
        dynamic_shapes=flat_dynamic_shapes,
    )

    metadata = metadata_for_compile(
        kmodel=kmodel,
        profile=profile,
        precision=precision,
        shapes=shapes,
        torch_tensorrt=torch_tensorrt,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
    )
    save_trt_metadata(output_dir, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write TensorRT artifact and metadata",
    )
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        help="Hugging Face model repo for config and PyTorch weights",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a Kokoro config.json file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Optional local PyTorch checkpoint path",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="TensorRT enabled precision",
    )
    parser.add_argument(
        "--workspace-size-mib",
        "--workspace_size_mib",
        type=int,
        default=512,
        help=(
            "TensorRT builder workspace cap in MiB. Lower values prevent the "
            "builder from trying very large tactics. Use 0 to omit this option "
            "for Torch-TensorRT versions that choose their own default."
        ),
    )
    parser.add_argument(
        "--builder-optimization-level",
        "--builder_optimization_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "TensorRT builder optimization level. Level 0 greatly reduces tactic "
            "search memory and is the safest default for Kokoro's long dynamic "
            "1D decoder/generator graph. Increase on larger GPUs if desired."
        ),
    )
    parser.add_argument(
        "--skip-profile-validation",
        "--skip_profile_validation",
        action="store_true",
        help="Skip PyTorch preflight validation of the TensorRT kMIN profile.",
    )
    parser.add_argument(
        "--min-frames",
        "--min_frames",
        type=int,
        default=2,
        help=(
            "Minimum synthesis-frame length in TensorRT profile. "
            "Must be high enough that f0/noise length is at least 3."
        ),
    )
    parser.add_argument(
        "--opt-frames",
        "--opt_frames",
        type=int,
        default=256,
        help="Optimized synthesis-frame length in TensorRT profile",
    )
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        type=int,
        default=1024,
        help="Maximum synthesis-frame length in TensorRT profile",
    )
    args = parser.parse_args()

    profile = TensorRTDynamicShapeProfile(
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
        max_frames=args.max_frames,
    )

    kmodel = KModel(
        repo_id=args.repo_id,
        config=args.config,
        model=args.model,
    ).eval()

    workspace_size = (
        None
        if args.workspace_size_mib <= 0
        else int(args.workspace_size_mib) * 1024 * 1024
    )

    compile_generator_with_source_pyramid(
        kmodel=kmodel,
        output_dir=args.output_dir,
        profile=profile,
        precision=args.precision,
        workspace_size=workspace_size,
        builder_optimization_level=args.builder_optimization_level,
        validate_profile=not args.skip_profile_validation,
    )


if __name__ == "__main__":
    main()
