import json
from pathlib import Path
from typing import Any, Optional, Union

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"

MODEL_FILENAMES: dict[str, str] = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}

CONFIG_FILENAME = "config.json"
ONNX_METADATA_FILENAME = "metadata.json"
TRT_METADATA_FILENAME = "metadata.json"

ONNX_TEXT_DURATION_PREFIX = "text_duration"
ONNX_ACOUSTIC_VOCODER_PREFIX = "acoustic_vocoder"


def resolve_repo_id(repo_id: Optional[str]) -> str:
    if repo_id is None:
        repo_id = DEFAULT_REPO_ID
        print(
            f"WARNING: Defaulting repo_id to {repo_id}. "
            f"Pass repo_id='{repo_id}' to suppress this warning."
        )
    return repo_id


def resolve_model_path(repo_id: str, model: Optional[Union[str, Path]] = None) -> str:
    if model is not None:
        return str(model)

    filename = MODEL_FILENAMES.get(repo_id)
    if filename is None:
        raise ValueError(
            f"No default checkpoint filename is known for repo_id={repo_id!r}. "
            "Pass a local model checkpoint path explicitly."
        )

    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_json(path: Union[str, Path]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as r:
        return json.load(r)


def save_json(path: Union[str, Path], data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as w:
        json.dump(data, w, indent=2, sort_keys=True)
        w.write("\n")


def load_config_data(
    repo_id: str,
    config: Union[dict[str, Any], str, Path, None] = None,
) -> dict[str, Any]:
    if isinstance(config, dict):
        return config

    config_path = config
    if config_path is None:
        config_path = hf_hub_download(repo_id=repo_id, filename=CONFIG_FILENAME)

    return load_json(config_path)


def get_context_length(config_data: dict[str, Any]) -> int:
    plbert = config_data.get("plbert", {})
    if isinstance(plbert, dict):
        return int(plbert.get("max_position_embeddings", 512))
    return 512


def load_exported_config(model_dir: Union[str, Path]) -> dict[str, Any]:
    return load_json(Path(model_dir) / CONFIG_FILENAME)


def save_exported_config(
    model_dir: Union[str, Path], config_data: dict[str, Any]
) -> None:
    save_json(Path(model_dir) / CONFIG_FILENAME, config_data)


def onnx_export_path(output_dir: Union[str, Path], prefix: str) -> Path:
    return Path(output_dir) / f"{prefix}.onnx"


def load_artifact_metadata(
    artifact_dir: Union[str, Path],
    filename: str = ONNX_METADATA_FILENAME,
) -> dict[str, Any]:
    return load_json(Path(artifact_dir) / filename)


def save_artifact_metadata(
    artifact_dir: Union[str, Path],
    metadata: dict[str, Any],
    filename: str = ONNX_METADATA_FILENAME,
) -> None:
    save_json(Path(artifact_dir) / filename, metadata)


def load_trt_metadata(artifact_dir: Union[str, Path]) -> dict[str, Any]:
    return load_artifact_metadata(artifact_dir, TRT_METADATA_FILENAME)


def save_trt_metadata(artifact_dir: Union[str, Path], metadata: dict[str, Any]) -> None:
    save_artifact_metadata(artifact_dir, metadata, TRT_METADATA_FILENAME)
