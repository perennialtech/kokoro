from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from .model import (
    END_SILENCE_FRAMES,
    KEEP_EOS_FRAMES,
    KModelOutput,
    ONNX_ACOUSTIC_VOCODER_PREFIX,
    ONNX_TEXT_DURATION_PREFIX,
    UtteranceOutput,
    expand_token_features,
    load_config_data,
    normalize_inference_inputs,
    onnx_export_path,
    resolve_repo_id,
)

ModelPath = Union[str, Path]


def _load_ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "KokoroONNXBackend requires onnxruntime. Install it with "
            "`pip install onnxruntime` or `pip install onnxruntime-gpu`."
        ) from e
    return ort


class KokoroONNXBackend:
    """
    Dynamic-shape ONNX Runtime inference backend.

    The backend executes the same canonical boundaries as the PyTorch backend:
    exact text sequence, host-side exact frame expansion, and one exact acoustic
    execution per utterance.
    """

    def __init__(
        self,
        text_duration: ModelPath,
        acoustic_vocoder: ModelPath,
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
    ):
        ort = _load_ort()
        self.providers = list(providers) if providers is not None else None
        self.session_options = session_options

        self.text_duration = ort.InferenceSession(
            str(text_duration),
            sess_options=self.session_options,
            providers=self.providers,
        )
        self.acoustic_vocoder = ort.InferenceSession(
            str(acoustic_vocoder),
            sess_options=self.session_options,
            providers=self.providers,
        )

    @classmethod
    def from_dir(
        cls,
        model_dir: Union[str, Path],
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
    ) -> "KokoroONNXBackend":
        model_dir = Path(model_dir)
        return cls(
            text_duration=onnx_export_path(model_dir, ONNX_TEXT_DURATION_PREFIX),
            acoustic_vocoder=onnx_export_path(model_dir, ONNX_ACOUSTIC_VOCODER_PREFIX),
            providers=providers,
            session_options=session_options,
        )

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> KModelOutput:
        batch = normalize_inference_inputs(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
            prepared=prepared,
        )

        input_ids_np = batch.input_ids.detach().cpu().numpy().astype(np.int64)
        input_lengths_np = batch.input_lengths.detach().cpu().numpy().astype(np.int64)
        ref_s_np = batch.ref_s.detach().cpu().numpy().astype(np.float32)
        speed_np = batch.speed.detach().cpu().numpy().astype(np.float32)

        utterances: list[UtteranceOutput] = []

        for b in range(input_ids_np.shape[0]):
            text_len = int(input_lengths_np[b])
            ids = np.ascontiguousarray(
                input_ids_np[b : b + 1, :text_len], dtype=np.int64
            )
            lengths = np.asarray([text_len], dtype=np.int64)
            ref = np.ascontiguousarray(ref_s_np[b : b + 1], dtype=np.float32)
            speed_item = np.ascontiguousarray(speed_np[b : b + 1], dtype=np.float32)

            duration_float_np, duration_hidden_np, text_hidden_np = (
                self.text_duration.run(
                    None,
                    {
                        "input_ids": ids,
                        "input_lengths": lengths,
                        "ref_s": ref,
                        "speed": speed_item,
                    },
                )
            )

            duration_float = torch.from_numpy(duration_float_np)
            duration_hidden = torch.from_numpy(duration_hidden_np)
            text_hidden = torch.from_numpy(text_hidden_np)
            input_lengths_t = torch.from_numpy(lengths)

            frames = expand_token_features(
                duration_float,
                duration_hidden,
                text_hidden,
                input_lengths_t,
                end_silence_frames=END_SILENCE_FRAMES,
                keep_eos_frames=KEEP_EOS_FRAMES,
            )
            item = frames.items[0]

            audio_np = self.acoustic_vocoder.run(
                None,
                {
                    "asr": np.ascontiguousarray(
                        item.asr.unsqueeze(0).numpy(),
                        dtype=np.float32,
                    ),
                    "en": np.ascontiguousarray(
                        item.en.unsqueeze(0).numpy(),
                        dtype=np.float32,
                    ),
                    "ref_s": ref,
                },
            )[0]

            samples_per_frame = audio_np.shape[-1] // item.synthesis_frame_length
            sample_length = item.return_frame_length * samples_per_frame
            audio = (
                torch.from_numpy(audio_np[..., :sample_length]).reshape(-1).contiguous()
            )

            utterances.append(
                UtteranceOutput(
                    audio=audio,
                    pred_dur=item.pred_dur,
                    duration_float=duration_float[0, :text_len].contiguous(),
                    synthesis_frame_length=item.synthesis_frame_length,
                    return_frame_length=item.return_frame_length,
                    sample_length=sample_length,
                    graphemes=batch.graphemes[b],
                    phonemes=batch.phonemes[b],
                )
            )

        return KModelOutput(utterances=utterances)


class KONNXModel:
    """
    Lightweight dynamic-ONNX counterpart to KModel.

    It owns model metadata needed by KPipeline and a KokoroONNXBackend for
    execution. Export with `KModel(...).export_onnx("onnx_dir")`, then load with
    `KONNXModel("onnx_dir")`.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        repo_id: Optional[str] = None,
        config: Union[dict[str, Any], str, Path, None] = None,
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
    ):
        self.repo_id = resolve_repo_id(repo_id)
        config_data = load_config_data(self.repo_id, config)

        plbert = config_data.get("plbert", {})
        self.vocab: Optional[dict[str, int]] = config_data.get("vocab")
        self.context_length = (
            int(plbert.get("max_position_embeddings", 512))
            if isinstance(plbert, dict)
            else 512
        )

        self.backend = KokoroONNXBackend.from_dir(
            model_dir,
            providers=providers,
            session_options=session_options,
        )

    def inference_backend(self):
        return self.backend

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> KModelOutput:
        return self.backend(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
            prepared=prepared,
        )
