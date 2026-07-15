from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import torch
from huggingface_hub import hf_hub_download

from .artifact import ArtifactMetadata, ArtifactPaths, TensorRTArtifact
from .config import save_json
from .model import GeneratorExportBuilder, KokoroModelLoader
from .onnx_export import export_generator_onnx
from .shapes import ShapePlan
from .trt_builder import build_engine_from_onnx

DEFAULT_ONNX_OPSET = 18


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


def validate_generator_shape_ranges(
    module: torch.nn.Module,
    model,
    plan: ShapePlan,
    dtype: torch.dtype,
    groups: tuple[str, ...] = ("lower",),
) -> None:
    shapes = plan.engine_shapes(model)
    input_order = plan.input_order()

    with torch.inference_mode():
        for group in groups:
            example_inputs = tuple(
                torch.zeros(
                    shapes[group][name],
                    device="cuda",
                    dtype=dtype,
                )
                for name in input_order
            )

            try:
                output = module(*example_inputs)
            except Exception as e:
                formatted_shapes = {name: shapes[group][name] for name in input_order}
                raise RuntimeError(
                    f"Generator shape point {group!r} is not self-consistent "
                    f"in PyTorch. Shapes: {formatted_shapes}"
                ) from e

            if output.dim() != 3 or output.shape[0] != 1 or output.shape[1] != 1:
                raise RuntimeError(
                    f"Generator shape point {group!r} returned unexpected output "
                    f"shape: {tuple(output.shape)}"
                )

            del output
            del example_inputs

    torch.cuda.empty_cache()


def expand_voice_args(voices: Optional[list[str]]) -> list[str]:
    if not voices:
        return []

    result: list[str] = []
    for group in voices:
        result.extend(v.strip() for v in group.split(",") if v.strip())
    return result


def copy_selected_voices(
    output_dir: Path,
    repo_id: str,
    voices: list[str],
) -> None:
    if not voices:
        return

    voice_dir = output_dir / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)

    for voice in voices:
        src_path = Path(voice)
        if src_path.exists():
            dst_name = src_path.name
        elif src_path.suffix == ".pt":
            raise FileNotFoundError(f"Voice file does not exist: {src_path}")
        else:
            src_path = Path(
                hf_hub_download(repo_id=repo_id, filename=f"voices/{voice}.pt")
            )
            dst_name = f"{voice}.pt"

        shutil.copyfile(src_path, voice_dir / dst_name)


def metadata_for_compile(
    *,
    model,
    precision: str,
    shapes: dict[str, dict[str, tuple[int, ...]]],
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
    onnx_opset: int,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> ArtifactMetadata:
    tensorrt_version = get_tensorrt_version()
    major, minor = torch.cuda.get_device_capability()
    return ArtifactMetadata.create(
        repo_id=model.repo_id,
        checkpoint={
            "path": model.model_path,
            "sha256": sha256_file(model.model_path),
        },
        gpu={
            "name": torch.cuda.get_device_name(),
            "compute_capability": f"sm_{major}{minor}",
        },
        versions={
            "torch": torch.__version__,
            "tensorrt": tensorrt_version,
        },
        precision=precision,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
        shapes=shapes,
        input_names=input_names,
        output_names=output_names,
        onnx_opset=onnx_opset,
        tensorrt_runtime_api={
            "runtime": "tensorrt.Runtime",
            "engine": "ICudaEngine",
            "context": "IExecutionContext",
            "execute": "execute_async_v3",
            "version": tensorrt_version,
        },
    )


def compile_artifact(
    output_dir: Union[str, Path],
    *,
    repo_id: Optional[str] = None,
    config: Union[dict[str, Any], str, Path, None] = None,
    model: Optional[Union[str, Path]] = None,
    precision: str = "fp32",
    workspace_size: Optional[int] = 512 * 1024 * 1024,
    builder_optimization_level: Optional[int] = 0,
    validate_profile: bool = True,
    include_voices: Optional[list[str]] = None,
    onnx_opset: int = DEFAULT_ONNX_OPSET,
) -> None:
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch-TensorRT compilation requires CUDA")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(output_dir)

    dtype = torch.float16 if precision == "fp16" else torch.float32
    loader = KokoroModelLoader(repo_id=repo_id, config=config, model=model)
    kokoro_model = loader.load(load_weights=True)

    plan = ShapePlan.from_model(kokoro_model)
    shapes = plan.engine_shapes(kokoro_model)

    save_json(paths.config_path, kokoro_model.config_data)
    kokoro_model.save_host_state(paths.host_state_path)
    copy_selected_voices(
        output_dir=output_dir,
        repo_id=kokoro_model.repo_id,
        voices=include_voices or [],
    )

    module = GeneratorExportBuilder.build(kokoro_model).to(device="cuda", dtype=dtype)

    if validate_profile:
        validate_generator_shape_ranges(
            module=module,
            model=kokoro_model,
            plan=plan,
            dtype=dtype,
        )

    example_inputs = plan.example_tensors(kokoro_model, dtype)

    export_generator_onnx(
        module=module,
        example_inputs=example_inputs,
        shape_plan=plan,
        model=kokoro_model,
        output_path=paths.onnx_path,
        opset_version=onnx_opset,
    )

    torch.cuda.empty_cache()

    build_engine_from_onnx(
        onnx_path=paths.onnx_path,
        engine_path=paths.engine_path,
        shapes=shapes,
        input_order=plan.input_order(),
        precision=precision,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
    )

    metadata = metadata_for_compile(
        model=kokoro_model,
        precision=precision,
        shapes=shapes,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
        onnx_opset=onnx_opset,
        input_names=plan.input_order(),
        output_names=("audio",),
    )
    TensorRTArtifact.write_metadata(paths.metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write TensorRT artifact",
    )
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        help="Hugging Face model repo for source config, weights, and selected voices",
    )
    parser.add_argument("--config", type=Path, help="Optional path to config.json")
    parser.add_argument("--model", type=str, help="Optional local PyTorch checkpoint")
    parser.add_argument(
        "--include-voice",
        "--include_voice",
        action="append",
        dest="include_voices",
        help="Voice name from source repo or local .pt path. Repeat or comma-separate.",
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
        help="TensorRT builder workspace cap in MiB. Use 0 to omit this option.",
    )
    parser.add_argument(
        "--builder-optimization-level",
        "--builder_optimization_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="TensorRT builder optimization level",
    )
    parser.add_argument(
        "--skip-profile-validation",
        "--skip_profile_validation",
        action="store_true",
        help="Skip PyTorch preflight validation of the automatic TensorRT shape range.",
    )
    parser.add_argument(
        "--onnx-opset", "--onnx_opset", type=int, default=DEFAULT_ONNX_OPSET
    )
    args = parser.parse_args()

    workspace_size = (
        None
        if args.workspace_size_mib <= 0
        else int(args.workspace_size_mib) * 1024 * 1024
    )

    compile_artifact(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        config=args.config,
        model=args.model,
        precision=args.precision,
        workspace_size=workspace_size,
        builder_optimization_level=args.builder_optimization_level,
        validate_profile=not args.skip_profile_validation,
        include_voices=expand_voice_args(args.include_voices),
        onnx_opset=args.onnx_opset,
    )


if __name__ == "__main__":
    main()
