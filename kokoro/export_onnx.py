"""ONNX PyTorch export wrapper."""

from pathlib import Path
from typing import Union

import torch

from .config import (
    ONNX_ACOUSTIC_VOCODER_PREFIX,
    ONNX_TEXT_DURATION_PREFIX,
    onnx_export_path,
    save_artifact_metadata,
    save_exported_config,
)
from .model import KModel


def export_onnx(model: KModel, output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.prepare_for_export()

    device = model.device

    text_duration = model.text_duration_module()
    acoustic_vocoder = model.acoustic_vocoder_module()

    input_ids = torch.zeros(1, 10, dtype=torch.long, device=device)
    ref_s = torch.zeros(1, 256, dtype=torch.float32, device=device)
    speed = torch.ones(1, dtype=torch.float32, device=device)

    text_duration_path = onnx_export_path(output_dir, ONNX_TEXT_DURATION_PREFIX)
    torch.onnx.export(
        text_duration,
        (input_ids, ref_s, speed),
        str(text_duration_path),
        input_names=["input_ids", "ref_s", "speed"],
        output_names=["duration_float", "duration_hidden", "text_hidden"],
        dynamic_axes={
            "input_ids": {1: "T"},
            "duration_float": {1: "T"},
            "duration_hidden": {1: "T"},
            "text_hidden": {1: "T"},
        },
        opset_version=26,
    )

    asr_channels = acoustic_vocoder.asr_channels
    en_channels = acoustic_vocoder.en_channels

    asr = torch.zeros(1, asr_channels, 10, dtype=torch.float32, device=device)
    en = torch.zeros(1, en_channels, 10, dtype=torch.float32, device=device)

    acoustic_vocoder_path = onnx_export_path(output_dir, ONNX_ACOUSTIC_VOCODER_PREFIX)
    torch.onnx.export(
        acoustic_vocoder,
        (asr, en, ref_s),
        str(acoustic_vocoder_path),
        input_names=["asr", "en", "ref_s"],
        output_names=["audio"],
        dynamic_axes={
            "asr": {2: "T_frames"},
            "en": {2: "T_frames"},
            "audio": {2: "sample_length"},
        },
        opset_version=26,
    )

    save_exported_config(output_dir, model.config_data)
    metadata = {
        "repo_id": model.repo_id,
    }
    save_artifact_metadata(output_dir, metadata)
