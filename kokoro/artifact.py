from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .config import load_json, save_json
from .shapes import Profile

CONFIG_FILENAME = "config.json"
TRT_METADATA_FILENAME = "metadata.json"
HOST_STATE_FILENAME = "host_state.pt"
TRT_ENGINE_FILENAME = "generator_with_source_pyramid.plan"
ONNX_FILENAME = "generator_with_source_pyramid.onnx"

ARTIFACT_TYPE = "kokoro_generator_with_source_pyramid_tensorrt_plan"
FORMAT_VERSION = 2


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.root / TRT_METADATA_FILENAME

    @property
    def host_state_path(self) -> Path:
        return self.root / HOST_STATE_FILENAME

    @property
    def engine_path(self) -> Path:
        return self.root / TRT_ENGINE_FILENAME

    @property
    def onnx_path(self) -> Path:
        return self.root / ONNX_FILENAME

    @property
    def voice_dir(self) -> Path:
        return self.root / "voices"


@dataclass(frozen=True)
class ArtifactMetadata:
    artifact_type: str
    format_version: int
    engine_file: str
    onnx_file: str
    config_file: str
    host_state_file: str
    repo_id: str
    checkpoint: dict[str, Any]
    gpu: dict[str, Any]
    versions: dict[str, Any]
    precision: str
    workspace_size: Optional[int]
    builder_optimization_level: Optional[int]
    profile: Profile
    shapes: dict[str, dict[str, tuple[int, ...]]]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    onnx_opset: int
    tensorrt_runtime_api: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        checkpoint: dict[str, Any],
        gpu: dict[str, Any],
        versions: dict[str, Any],
        precision: str,
        workspace_size: Optional[int],
        builder_optimization_level: Optional[int],
        profile: Profile,
        shapes: dict[str, dict[str, tuple[int, ...]]],
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        onnx_opset: int,
        tensorrt_runtime_api: dict[str, Any],
    ) -> "ArtifactMetadata":
        metadata = cls(
            artifact_type=ARTIFACT_TYPE,
            format_version=FORMAT_VERSION,
            engine_file=TRT_ENGINE_FILENAME,
            onnx_file=ONNX_FILENAME,
            config_file=CONFIG_FILENAME,
            host_state_file=HOST_STATE_FILENAME,
            repo_id=repo_id,
            checkpoint=checkpoint,
            gpu=gpu,
            versions=versions,
            precision=precision,
            workspace_size=workspace_size,
            builder_optimization_level=builder_optimization_level,
            profile=profile,
            shapes=shapes,
            input_names=tuple(input_names),
            output_names=tuple(output_names),
            onnx_opset=int(onnx_opset),
            tensorrt_runtime_api=dict(tensorrt_runtime_api),
        )
        metadata.validate()
        return metadata

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactMetadata":
        required = {
            "artifact_type",
            "format_version",
            "engine_file",
            "onnx_file",
            "config_file",
            "host_state_file",
            "repo_id",
            "checkpoint",
            "gpu",
            "versions",
            "precision",
            "profile",
            "shapes",
            "input_names",
            "output_names",
            "onnx_opset",
            "tensorrt_runtime_api",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"TensorRT artifact metadata is missing keys: {missing}")

        profile = Profile.from_dict(data["profile"])
        shapes = cls._parse_shapes(data["shapes"])

        metadata = cls(
            artifact_type=str(data["artifact_type"]),
            format_version=int(data["format_version"]),
            engine_file=str(data["engine_file"]),
            onnx_file=str(data["onnx_file"]),
            config_file=str(data["config_file"]),
            host_state_file=str(data["host_state_file"]),
            repo_id=str(data["repo_id"]),
            checkpoint=dict(data["checkpoint"]),
            gpu=dict(data["gpu"]),
            versions=dict(data["versions"]),
            precision=str(data["precision"]).lower(),
            workspace_size=(
                None
                if data.get("workspace_size") is None
                else int(data["workspace_size"])
            ),
            builder_optimization_level=(
                None
                if data.get("builder_optimization_level") is None
                else int(data["builder_optimization_level"])
            ),
            profile=profile,
            shapes=shapes,
            input_names=tuple(str(name) for name in data["input_names"]),
            output_names=tuple(str(name) for name in data["output_names"]),
            onnx_opset=int(data["onnx_opset"]),
            tensorrt_runtime_api=dict(data["tensorrt_runtime_api"]),
        )
        metadata.validate()
        return metadata

    @staticmethod
    def _parse_shapes(raw: Any) -> dict[str, dict[str, tuple[int, ...]]]:
        if not isinstance(raw, dict):
            raise ValueError("metadata.shapes must be an object")

        if set(raw) != {"min", "opt", "max"}:
            raise ValueError("metadata.shapes must contain exactly min, opt, and max")

        parsed: dict[str, dict[str, tuple[int, ...]]] = {}
        for group in ("min", "opt", "max"):
            specs = raw[group]
            if not isinstance(specs, dict):
                raise ValueError(f"metadata.shapes.{group} must be an object")

            if "x" not in specs or "ref_s" not in specs:
                raise ValueError(f"metadata.shapes.{group} must contain x and ref_s")

            source_names = [name for name in specs if name.startswith("source_")]
            if not source_names:
                raise ValueError(
                    f"metadata.shapes.{group} must contain at least one source tensor"
                )

            parsed[group] = {}
            for name, shape in specs.items():
                if not isinstance(shape, (list, tuple)):
                    raise ValueError(f"metadata.shapes.{group}.{name} must be a list")

                dims = tuple(int(dim) for dim in shape)
                if not dims or any(dim <= 0 for dim in dims):
                    raise ValueError(
                        f"metadata.shapes.{group}.{name} must contain positive dims"
                    )
                parsed[group][name] = dims

        return parsed

    def validate(self) -> None:
        if self.artifact_type != ARTIFACT_TYPE:
            raise ValueError(
                f"Unsupported artifact_type {self.artifact_type!r}; expected {ARTIFACT_TYPE!r}"
            )
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported artifact format_version {self.format_version}; expected {FORMAT_VERSION}"
            )
        if self.engine_file != TRT_ENGINE_FILENAME:
            raise ValueError(
                f"Unsupported engine_file {self.engine_file!r}; expected {TRT_ENGINE_FILENAME!r}"
            )
        if not self.engine_file.endswith(".plan"):
            raise ValueError(
                "metadata.engine_file must be a native TensorRT .plan file"
            )
        if self.onnx_file != ONNX_FILENAME:
            raise ValueError(
                f"Unsupported onnx_file {self.onnx_file!r}; expected {ONNX_FILENAME!r}"
            )
        if not self.onnx_file.endswith(".onnx"):
            raise ValueError("metadata.onnx_file must be an ONNX file")
        if self.config_file != CONFIG_FILENAME:
            raise ValueError(
                f"Unsupported config_file {self.config_file!r}; expected {CONFIG_FILENAME!r}"
            )
        if self.host_state_file != HOST_STATE_FILENAME:
            raise ValueError(
                f"Unsupported host_state_file {self.host_state_file!r}; expected {HOST_STATE_FILENAME!r}"
            )
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("metadata.precision must be fp32 or fp16")
        if not self.repo_id:
            raise ValueError("metadata.repo_id must not be empty")

        expected_cc = self.gpu.get("compute_capability")
        if not isinstance(expected_cc, str) or not expected_cc.startswith("sm_"):
            raise ValueError("metadata.gpu.compute_capability must look like sm_89")

        self.profile.validate()

        if not self.input_names:
            raise ValueError("metadata.input_names must not be empty")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("metadata.input_names must be unique")
        if self.input_names[0:2] != ("x", "ref_s"):
            raise ValueError("metadata.input_names must start with x and ref_s")

        source_names = tuple(
            name for name in self.input_names if name.startswith("source_")
        )
        if not source_names:
            raise ValueError("metadata.input_names must contain source_i inputs")

        for group, specs in self.shapes.items():
            missing = [name for name in self.input_names if name not in specs]
            if missing:
                raise ValueError(
                    f"metadata.shapes.{group} is missing inputs: {missing}"
                )

        if self.output_names != ("audio",):
            raise ValueError("metadata.output_names must be exactly ['audio']")
        if self.onnx_opset <= 0:
            raise ValueError("metadata.onnx_opset must be positive")
        if not self.tensorrt_runtime_api:
            raise ValueError("metadata.tensorrt_runtime_api must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "format_version": self.format_version,
            "engine_file": self.engine_file,
            "onnx_file": self.onnx_file,
            "config_file": self.config_file,
            "host_state_file": self.host_state_file,
            "repo_id": self.repo_id,
            "checkpoint": self.checkpoint,
            "gpu": self.gpu,
            "versions": self.versions,
            "precision": self.precision,
            "workspace_size": self.workspace_size,
            "builder_optimization_level": self.builder_optimization_level,
            "profile": self.profile.to_dict(),
            "shapes": {
                group: {name: list(shape) for name, shape in specs.items()}
                for group, specs in self.shapes.items()
            },
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "onnx_opset": self.onnx_opset,
            "tensorrt_runtime_api": self.tensorrt_runtime_api,
        }


@dataclass(frozen=True)
class TensorRTArtifact:
    paths: ArtifactPaths
    metadata: ArtifactMetadata

    @classmethod
    def load(cls, root: Union[str, Path]) -> "TensorRTArtifact":
        paths = ArtifactPaths(Path(root))
        if not paths.metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing TensorRT artifact metadata: {paths.metadata_path}"
            )

        metadata = ArtifactMetadata.from_dict(load_json(paths.metadata_path))
        artifact = cls(paths=paths, metadata=metadata)
        artifact.validate_files()
        return artifact

    def validate_files(self) -> None:
        missing = [
            path
            for path in (
                self.paths.config_path,
                self.paths.host_state_path,
                self.paths.engine_path,
                self.paths.onnx_path,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "TensorRT artifact is incomplete; missing: "
                + ", ".join(str(path) for path in missing)
            )

    def validate_gpu(self) -> None:
        import torch

        expected = self.metadata.gpu["compute_capability"]
        major, minor = torch.cuda.get_device_capability()
        actual = f"sm_{major}{minor}"

        if expected != actual:
            raise RuntimeError(
                "TensorRT artifact was compiled for GPU compute capability "
                f"{expected}, but current GPU is {actual}. Recompile the artifact "
                "on this GPU or another GPU with the same compute capability."
            )

    def save_config(self, config_data: dict[str, Any]) -> None:
        save_json(self.paths.config_path, config_data)

    def save_host_state(self, model) -> None:
        model.save_host_state(self.paths.host_state_path)

    def save_metadata(self) -> None:
        save_json(self.paths.metadata_path, self.metadata.to_dict())

    def load_config(self) -> dict[str, Any]:
        return load_json(self.paths.config_path)

    @staticmethod
    def write_metadata(path: Union[str, Path], metadata: ArtifactMetadata) -> None:
        with open(path, "w", encoding="utf-8") as w:
            json.dump(metadata.to_dict(), w, indent=2, sort_keys=True)
            w.write("\n")
