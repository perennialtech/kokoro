from typing import Any

import torch

from .types import END_SILENCE_FRAMES, KEEP_EOS_FRAMES, FrameItem, SynthesisResult


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


@torch.inference_mode()
def synthesize_prepared_trt(tts: Any, prepared: Any) -> SynthesisResult:
    input_ids = prepared.input_ids
    ref_s = prepared.ref_s
    speed = prepared.speed

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"prepared.input_ids must have canonical shape [1,T], got {tuple(input_ids.shape)}"
        )
    if ref_s.dim() != 2 or ref_s.shape != (1, 256):
        raise ValueError(
            f"prepared.ref_s must have canonical shape [1,256], got {tuple(ref_s.shape)}"
        )
    if speed.dim() != 1 or speed.shape[0] != 1:
        raise ValueError(
            f"prepared.speed must have canonical shape [1], got {tuple(speed.shape)}"
        )

    input_ids = input_ids.contiguous().to(device=tts.device, dtype=torch.long)
    ref_s = ref_s.contiguous().to(device=tts.device, dtype=torch.float32)
    speed = speed.contiguous().to(device=tts.device, dtype=torch.float32)

    duration_float, duration_hidden, text_hidden = tts.host.text_duration(
        input_ids,
        ref_s,
        speed,
    )

    frame_item = expand_frames(
        duration_float,
        duration_hidden,
        text_hidden,
        end_silence_frames=END_SILENCE_FRAMES,
        keep_eos_frames=KEEP_EOS_FRAMES,
    )

    audio = tts.render_frame(frame_item, ref_s)
    samples_per_frame = audio.shape[-1] // frame_item.synthesis_frame_length
    sample_length = frame_item.return_frame_length * samples_per_frame
    audio = audio[..., :sample_length].reshape(-1).contiguous()

    return SynthesisResult(
        audio=audio,
        pred_dur=frame_item.pred_dur,
        duration_float=duration_float[0, : prepared.input_length].contiguous(),
        synthesis_frame_length=frame_item.synthesis_frame_length,
        return_frame_length=frame_item.return_frame_length,
        sample_length=sample_length,
        graphemes=getattr(prepared, "graphemes", None),
        phonemes=getattr(prepared, "phonemes", None),
    )
