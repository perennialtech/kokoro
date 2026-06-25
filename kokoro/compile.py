from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import torch
from huggingface_hub import hf_hub_download

from .artifact import ArtifactMetadata, ArtifactPaths, TensorRTArtifact
from .config import save_json
from .model import GeneratorExportBuilder, KokoroHostStages, KokoroModelLoader
from .shapes import Profile, ShapePlan

SUPPORTED_TORCH_TENSORRT = (2, 12, 1)


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


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return (0, 0, 0)
    return tuple(int(group) for group in match.groups())


def require_torch_tensorrt():
    try:
        import torch_tensorrt
    except ImportError as e:
        raise ImportError(
            "TensorRT compilation requires torch-tensorrt >= 2.12.1. "
            "Install Torch-TensorRT matching your PyTorch/CUDA/TensorRT stack."
        ) from e

    current = version_tuple(str(getattr(torch_tensorrt, "__version__", "0.0.0")))
    if current < SUPPORTED_TORCH_TENSORRT:
        raise RuntimeError(
            "Kokoro TensorRT compilation supports torch-tensorrt >= 2.12.1 only; "
            f"found {getattr(torch_tensorrt, '__version__', None)!r}"
        )

    return torch_tensorrt


def get_tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt

        return str(trt.__version__)
    except Exception:
        return None


def compile_exported_program_with_tensorrt(
    torch_tensorrt: Any,
    exported_program: torch.export.ExportedProgram,
    inputs: list[Any],
    enabled_precisions: set[torch.dtype],
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
) -> Any:
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
        kwargs["workspace_size"] = workspace_size

    if builder_optimization_level is not None:
        builder_optimization_level = int(builder_optimization_level)
        if not 0 <= builder_optimization_level <= 5:
            raise ValueError("builder_optimization_level must be between 0 and 5")
        kwargs["optimization_level"] = builder_optimization_level

    return torch_tensorrt.compile(exported_program, **kwargs)


def validate_generator_profile_shapes(
    module: torch.nn.Module,
    model,
    plan: ShapePlan,
    dtype: torch.dtype,
    groups: tuple[str, ...] = ("min",),
) -> None:
    shapes = plan.profile_shapes(model)
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
                    f"Generator profile point {group!r} is not self-consistent "
                    f"in PyTorch. Shapes: {formatted_shapes}"
                ) from e

            if output.dim() != 3 or output.shape[0] != 1 or output.shape[1] != 1:
                raise RuntimeError(
                    f"Generator profile point {group!r} returned unexpected output "
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
    profile: Profile,
    precision: str,
    shapes: dict[str, dict[str, tuple[int, ...]]],
    torch_tensorrt: Any,
    workspace_size: Optional[int],
    builder_optimization_level: Optional[int],
) -> ArtifactMetadata:
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
            "tensorrt": get_tensorrt_version(),
            "torch_tensorrt": str(getattr(torch_tensorrt, "__version__", None)),
        },
        precision=precision,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
        profile=profile,
        shapes=shapes,
    )


def compile_artifact(
    output_dir: Union[str, Path],
    *,
    repo_id: Optional[str] = None,
    config: Union[dict[str, Any], str, Path, None] = None,
    model: Optional[Union[str, Path]] = None,
    profile: Optional[Profile] = None,
    precision: str = "fp32",
    workspace_size: Optional[int] = 512 * 1024 * 1024,
    builder_optimization_level: Optional[int] = 0,
    validate_profile: bool = True,
    include_voices: Optional[list[str]] = None,
) -> None:
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch-TensorRT compilation requires CUDA")

    torch_tensorrt = require_torch_tensorrt()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(output_dir)

    profile = profile or Profile(min_frames=2, opt_frames=256, max_frames=1024)
    profile.validate()

    dtype = torch.float16 if precision == "fp16" else torch.float32
    loader = KokoroModelLoader(repo_id=repo_id, config=config, model=model)
    kokoro_model = loader.load(load_weights=True)

    plan = ShapePlan.from_model(kokoro_model, profile)
    shapes = plan.profile_shapes(kokoro_model)

    save_json(paths.config_path, kokoro_model.config_data)
    kokoro_model.save_host_state(paths.host_state_path)
    copy_selected_voices(
        output_dir=output_dir,
        repo_id=kokoro_model.repo_id,
        voices=include_voices or [],
    )

    module = GeneratorExportBuilder.build(kokoro_model).to(device="cuda", dtype=dtype)

    if validate_profile:
        validate_generator_profile_shapes(
            module=module,
            model=kokoro_model,
            plan=plan,
            dtype=dtype,
            groups=("min",),
        )

    inputs = plan.tensorrt_inputs(
        model=kokoro_model,
        dtype=dtype,
        torch_tensorrt=torch_tensorrt,
    )
    example_inputs = plan.example_tensors(kokoro_model, dtype)
    dynamic_shapes = plan.export_dynamic_shapes(kokoro_model)
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

    torch_tensorrt.save(
        compiled,
        str(paths.engine_path),
        arg_inputs=example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    metadata = metadata_for_compile(
        model=kokoro_model,
        profile=profile,
        precision=precision,
        shapes=shapes,
        torch_tensorrt=torch_tensorrt,
        workspace_size=workspace_size,
        builder_optimization_level=builder_optimization_level,
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
        help="Skip PyTorch preflight validation of TensorRT kMIN profile.",
    )
    parser.add_argument("--min-frames", "--min_frames", type=int, default=2)
    parser.add_argument("--opt-frames", "--opt_frames", type=int, default=256)
    parser.add_argument("--max-frames", "--max_frames", type=int, default=1024)
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
        profile=Profile(
            min_frames=args.min_frames,
            opt_frames=args.opt_frames,
            max_frames=args.max_frames,
        ),
        precision=args.precision,
        workspace_size=workspace_size,
        builder_optimization_level=args.builder_optimization_level,
        validate_profile=not args.skip_profile_validation,
        include_voices=expand_voice_args(args.include_voices),
    )


if __name__ == "__main__":
    main()
