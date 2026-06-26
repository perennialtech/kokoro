from dataclasses import dataclass
from typing import Optional

import torch

from .telemetry import ChunkTrace

END_SILENCE_FRAMES = 5
KEEP_EOS_FRAMES = 1


@dataclass
class FrameItem:
    asr: torch.Tensor
    en: torch.Tensor
    pred_dur: torch.Tensor
    synthesis_frame_length: int
    return_frame_length: int


@dataclass
class SynthesisResult:
    audio: torch.Tensor
    pred_dur: torch.Tensor
    duration_float: torch.Tensor
    synthesis_frame_length: int
    return_frame_length: int
    sample_length: int
    graphemes: Optional[str] = None
    phonemes: Optional[str] = None
    profile: Optional[ChunkTrace] = None
