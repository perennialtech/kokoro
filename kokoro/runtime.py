from typing import Optional

import torch

from .types import (END_SILENCE_FRAMES, KEEP_EOS_FRAMES, FrameItem,
                    InferenceRequest, KModelOutput, UtteranceOutput)


def _as_tensor(value, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _move(
    tensor: torch.Tensor,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None and dtype is None:
        return tensor
    return tensor.to(
        device=device if device is not None else tensor.device,
        dtype=dtype if dtype is not None else tensor.dtype,
    )


def normalize_requests(
    input_ids: Optional[torch.Tensor] = None,
    input_lengths: Optional[torch.Tensor] = None,
    ref_s: Optional[torch.Tensor] = None,
    speed: Optional[torch.Tensor] = None,
    prepared=None,
    device: Optional[torch.device] = None,
    ref_dtype: torch.dtype = torch.float32,
) -> list[InferenceRequest]:
    if prepared is not None:
        input_ids = _as_tensor(prepared.input_ids, torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"prepared.input_ids must have shape [T] or [1,T], got {tuple(input_ids.shape)}"
            )

        length = int(prepared.input_length)
        if length <= 0:
            raise ValueError("prepared.input_length must be positive")

        ref = _as_tensor(prepared.ref_s, ref_dtype)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)

        speed_tensor = torch.tensor([float(prepared.speed)], dtype=torch.float32)

        return [
            InferenceRequest(
                input_ids=_move(input_ids[:, :length].contiguous(), device),
                ref_s=_move(ref.contiguous(), device, ref_dtype),
                speed=_move(speed_tensor, device),
                graphemes=getattr(prepared, "graphemes", None),
                phonemes=getattr(prepared, "phonemes", None),
            )
        ]

    if input_ids is None or ref_s is None or speed is None:
        raise ValueError(
            "input_ids, ref_s, and speed are required unless prepared is provided"
        )

    input_ids = _as_tensor(input_ids, torch.long)
    ref_s = _as_tensor(ref_s, ref_dtype)
    speed = _as_tensor(speed, torch.float32)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    if speed.dim() == 0:
        speed = speed.unsqueeze(0)

    if input_ids.dim() != 2:
        raise ValueError(
            f"Expected input_ids [T] or [B,T], got {tuple(input_ids.shape)}"
        )
    if ref_s.dim() != 2:
        raise ValueError(f"Expected ref_s [256] or [B,256], got {tuple(ref_s.shape)}")
    if speed.dim() != 1:
        raise ValueError(f"Expected speed scalar or [B], got {tuple(speed.shape)}")

    batch_size = int(input_ids.shape[0])

    if input_lengths is None:
        lengths = torch.full((batch_size,), input_ids.shape[1], dtype=torch.long)
    else:
        lengths = _as_tensor(input_lengths, torch.long)
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

    if lengths.numel() != batch_size:
        raise ValueError(
            f"input_lengths must have {batch_size} entries, got {lengths.numel()}"
        )

    if ref_s.shape[0] not in (1, batch_size):
        raise ValueError(f"ref_s batch must be 1 or {batch_size}, got {ref_s.shape[0]}")
    if speed.shape[0] not in (1, batch_size):
        raise ValueError(f"speed batch must be 1 or {batch_size}, got {speed.shape[0]}")

    requests: list[InferenceRequest] = []
    max_len = int(input_ids.shape[1])

    for b in range(batch_size):
        length = int(lengths[b].item())
        if length <= 0:
            raise ValueError("input_lengths entries must be positive")
        if length > max_len:
            raise ValueError(f"input length {length} exceeds input_ids width {max_len}")

        ref_index = b if ref_s.shape[0] == batch_size else 0
        speed_index = b if speed.shape[0] == batch_size else 0

        requests.append(
            InferenceRequest(
                input_ids=_move(input_ids[b : b + 1, :length].contiguous(), device),
                ref_s=_move(
                    ref_s[ref_index : ref_index + 1].contiguous(), device, ref_dtype
                ),
                speed=_move(speed[speed_index : speed_index + 1].contiguous(), device),
            )
        )

    return requests


def expand_frames(
    duration_float: torch.Tensor,
    duration_hidden: torch.Tensor,
    text_hidden: torch.Tensor,
    end_silence_frames: int = END_SILENCE_FRAMES,
    keep_eos_frames: int = KEEP_EOS_FRAMES,
) -> FrameItem:
    if duration_float.dim() != 2 or duration_float.shape[0] != 1:
        raise ValueError(
            f"duration_float must have shape [1,T], got {tuple(duration_float.shape)}"
        )

    n = int(duration_float.shape[1])
    if n <= 0:
        raise ValueError("duration_float must contain at least one token")

    d = torch.round(duration_float[0]).clamp(min=1).long()

    if end_silence_frames > 0:
        d[-1] += end_silence_frames

    idx = torch.repeat_interleave(torch.arange(n, device=duration_float.device), d)
    synthesis_frame_length = int(idx.numel())

    asr = text_hidden[0, idx, :].transpose(0, 1).contiguous()
    en = duration_hidden[0, idx, :].transpose(0, 1).contiguous()

    eos_frames = int(d[-1].item())
    trim_frames = max(eos_frames - keep_eos_frames, 0)
    return_frame_length = max(1, synthesis_frame_length - trim_frames)

    return FrameItem(
        asr=asr,
        en=en,
        pred_dur=d,
        synthesis_frame_length=synthesis_frame_length,
        return_frame_length=return_frame_length,
    )


class Synthesizer:
    def __init__(self, backend):
        self.backend = backend

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> KModelOutput:
        device = getattr(self.backend, "device", None)
        ref_dtype = getattr(self.backend, "preferred_ref_dtype", torch.float32)

        requests = normalize_requests(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
            prepared=prepared,
            device=device,
            ref_dtype=ref_dtype,
        )

        utterances: list[UtteranceOutput] = []

        for request in requests:
            duration_float, duration_hidden, text_hidden = self.backend.text_duration(
                request.input_ids,
                request.ref_s,
                request.speed,
            )

            frame_item = expand_frames(
                duration_float,
                duration_hidden,
                text_hidden,
                end_silence_frames=END_SILENCE_FRAMES,
                keep_eos_frames=KEEP_EOS_FRAMES,
            )

            audio = self.backend.render(frame_item, request.ref_s)
            samples_per_frame = audio.shape[-1] // frame_item.synthesis_frame_length
            sample_length = frame_item.return_frame_length * samples_per_frame
            audio = audio[..., :sample_length].reshape(-1).contiguous()

            utterances.append(
                UtteranceOutput(
                    audio=audio,
                    pred_dur=frame_item.pred_dur,
                    duration_float=duration_float[
                        0, : request.input_length
                    ].contiguous(),
                    synthesis_frame_length=frame_item.synthesis_frame_length,
                    return_frame_length=frame_item.return_frame_length,
                    sample_length=sample_length,
                    graphemes=request.graphemes,
                    phonemes=request.phonemes,
                )
            )

        return KModelOutput(utterances=utterances)
