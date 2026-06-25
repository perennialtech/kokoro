from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

from .config import (ONNX_ACOUSTIC_VOCODER_PREFIX, ONNX_TEXT_DURATION_PREFIX,
                     get_context_length, load_artifact_metadata,
                     load_exported_config, onnx_export_path)
from .runtime import Synthesizer
from .types import FrameItem, KModelOutput

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
    Dynamic-shape ONNX Runtime backend using the shared exact synthesis runtime.
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
        self.preferred_ref_device = None
        self.preferred_ref_dtype = torch.float32

        self.text_duration_session = ort.InferenceSession(
            str(text_duration),
            sess_options=self.session_options,
            providers=self.providers,
        )
        self.acoustic_vocoder_session = ort.InferenceSession(
            str(acoustic_vocoder),
            sess_options=self.session_options,
            providers=self.providers,
        )
        self.synthesizer = Synthesizer(self)

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

    def text_duration(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ):
        duration_float_np, duration_hidden_np, text_hidden_np = (
            self.text_duration_session.run(
                None,
                {
                    "input_ids": np.ascontiguousarray(
                        input_ids.detach().cpu().numpy(),
                        dtype=np.int64,
                    ),
                    "ref_s": np.ascontiguousarray(
                        ref_s.detach().cpu().numpy(),
                        dtype=np.float32,
                    ),
                    "speed": np.ascontiguousarray(
                        speed.detach().cpu().numpy(),
                        dtype=np.float32,
                    ),
                },
            )
        )

        return (
            torch.from_numpy(duration_float_np),
            torch.from_numpy(duration_hidden_np),
            torch.from_numpy(text_hidden_np),
        )

    def render(self, frame_item: FrameItem, ref_s: torch.Tensor) -> torch.Tensor:
        audio_np = self.acoustic_vocoder_session.run(
            None,
            {
                "asr": np.ascontiguousarray(
                    frame_item.asr.unsqueeze(0).detach().cpu().numpy(),
                    dtype=np.float32,
                ),
                "en": np.ascontiguousarray(
                    frame_item.en.unsqueeze(0).detach().cpu().numpy(),
                    dtype=np.float32,
                ),
                "ref_s": np.ascontiguousarray(
                    ref_s.detach().cpu().numpy(),
                    dtype=np.float32,
                ),
            },
        )[0]

        return torch.from_numpy(audio_np)

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> KModelOutput:
        return self.synthesizer(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
            prepared=prepared,
        )


class KONNXModel:
    """
    Self-contained ONNX counterpart to KModel.

    The ONNX directory must contain:
      - text_duration.onnx
      - acoustic_vocoder.onnx
      - config.json
      - metadata.json
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
    ):
        self.model_dir = Path(model_dir)
        self.metadata = load_artifact_metadata(self.model_dir)
        self.config_data = load_exported_config(self.model_dir)

        self.repo_id = self.metadata["repo_id"]
        self.vocab: Optional[dict[str, int]] = self.config_data.get("vocab")
        self.context_length = get_context_length(self.config_data)

        self.backend = KokoroONNXBackend.from_dir(
            self.model_dir,
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
