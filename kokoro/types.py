from dataclasses import dataclass
from typing import Optional

import torch

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
class UtteranceOutput:
    audio: torch.Tensor
    pred_dur: torch.Tensor
    duration_float: torch.Tensor
    synthesis_frame_length: int
    return_frame_length: int
    sample_length: int
    graphemes: Optional[str] = None
    phonemes: Optional[str] = None


@dataclass
class KModelOutput:
    utterances: list[UtteranceOutput]

    def to_padded_audio(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.utterances:
            return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)

        sample_lengths = torch.tensor(
            [u.sample_length for u in self.utterances],
            dtype=torch.long,
            device=self.utterances[0].audio.device,
        )
        max_len = int(sample_lengths.max().item())
        audio = self.utterances[0].audio.new_zeros((len(self.utterances), max_len))

        for i, utterance in enumerate(self.utterances):
            audio[i, : utterance.sample_length] = utterance.audio[
                : utterance.sample_length
            ]

        return audio, sample_lengths


@dataclass
class InferenceRequest:
    input_ids: torch.Tensor
    ref_s: torch.Tensor
    speed: torch.Tensor
    graphemes: Optional[str] = None
    phonemes: Optional[str] = None

    @property
    def input_length(self) -> int:
        return int(self.input_ids.shape[1])
