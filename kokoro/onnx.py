from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union
import re

import numpy as np
import torch

from .model import (
    DEFAULT_FRAME_BUCKETS,
    DEFAULT_TEXT_BUCKETS,
    KModel,
    ONNX_ACOUSTIC_VOCODER_PREFIX,
    ONNX_TEXT_DURATION_PREFIX,
    expand_token_features,
    load_config_data,
    resolve_repo_id,
)

ModelPath = Union[str, Path]
ModelSpec = Union[ModelPath, Mapping[int, ModelPath]]


def _load_ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "KokoroONNXBackend requires onnxruntime. Install it with "
            "`pip install onnxruntime` or `pip install onnxruntime-gpu`."
        ) from e
    return ort


def _as_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _infer_bucket(session: Any, axis: int) -> Optional[int]:
    shape = session.get_inputs()[0].shape
    if len(shape) <= axis:
        return None

    dim = shape[axis]
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str) and dim.isdigit():
        return int(dim)
    return None


def _discover_models(directory: Path, prefix: str) -> ModelSpec:
    models: dict[int, Path] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.onnx$")

    for path in sorted(directory.glob(f"{prefix}_*.onnx")):
        match = pattern.match(path.name)
        if match:
            models[int(match.group(1))] = path

    if models:
        return models

    single = directory / f"{prefix}.onnx"
    if single.exists():
        return single

    raise FileNotFoundError(f"No {prefix} ONNX models found in {directory}")


class KokoroONNXBackend:
    """
    ONNX Runtime inference backend.

    It mirrors KokoroInferenceBackend: feed tensors produced by KPipeline, or pass
    a KPipeline.PreparedInput via `prepared=...`. Text duration and acoustic
    vocoder are intentionally exported as bucketed models for predictable latency
    and memory use.
    """

    def __init__(
        self,
        text_duration: ModelSpec,
        acoustic_vocoder: ModelSpec,
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
        frame_buckets: Optional[Sequence[int]] = None,
    ):
        ort = _load_ort()
        self.providers = list(providers) if providers is not None else None
        self.session_options = session_options

        self.text_duration = self._load_sessions(
            ort, text_duration, bucket_axis=1, label=ONNX_TEXT_DURATION_PREFIX
        )
        self.acoustic_vocoder = self._load_sessions(
            ort, acoustic_vocoder, bucket_axis=2, label=ONNX_ACOUSTIC_VOCODER_PREFIX
        )

        text_buckets = sorted(k for k in self.text_duration if k is not None)
        acoustic_buckets = sorted(k for k in self.acoustic_vocoder if k is not None)

        self.text_buckets = tuple(text_buckets) or DEFAULT_TEXT_BUCKETS
        self.frame_buckets = tuple(
            frame_buckets or acoustic_buckets or DEFAULT_FRAME_BUCKETS
        )

    @classmethod
    def from_dir(
        cls,
        model_dir: Union[str, Path],
        providers: Optional[Sequence[Any]] = None,
        session_options: Optional[Any] = None,
        frame_buckets: Optional[Sequence[int]] = None,
    ) -> "KokoroONNXBackend":
        model_dir = Path(model_dir)
        return cls(
            text_duration=_discover_models(model_dir, ONNX_TEXT_DURATION_PREFIX),
            acoustic_vocoder=_discover_models(model_dir, ONNX_ACOUSTIC_VOCODER_PREFIX),
            providers=providers,
            session_options=session_options,
            frame_buckets=frame_buckets,
        )

    def _load_sessions(
        self,
        ort: Any,
        spec: ModelSpec,
        bucket_axis: int,
        label: str,
    ) -> dict[Optional[int], Any]:
        items = spec.items() if isinstance(spec, Mapping) else [(None, spec)]
        sessions: dict[Optional[int], Any] = {}

        for requested_bucket, path in items:
            session = ort.InferenceSession(
                str(path),
                sess_options=self.session_options,
                providers=self.providers,
            )

            inferred_bucket = _infer_bucket(session, bucket_axis)
            bucket = (
                int(requested_bucket)
                if requested_bucket is not None
                else inferred_bucket
            )

            if (
                requested_bucket is not None
                and inferred_bucket is not None
                and int(requested_bucket) != inferred_bucket
            ):
                raise ValueError(
                    f"{label} model {path} was registered for bucket "
                    f"{requested_bucket}, but its input shape is bucket {inferred_bucket}"
                )

            if bucket in sessions:
                raise ValueError(f"Duplicate {label} ONNX model for bucket {bucket}")

            sessions[bucket] = session

        if not sessions:
            raise ValueError(f"No {label} ONNX sessions were provided")

        return sessions

    @staticmethod
    def _session_for(sessions: Mapping[Optional[int], Any], bucket: int, label: str):
        if bucket in sessions:
            return sessions[bucket]
        if None in sessions and len(sessions) == 1:
            return sessions[None]

        available = sorted(k for k in sessions if k is not None)
        raise ValueError(
            f"No {label} ONNX model for bucket {bucket}. "
            f"Available buckets: {available}"
        )

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> "KModel.Output":
        if prepared is not None:
            input_ids = prepared.input_ids
            input_lengths = prepared.input_lengths
            ref_s = prepared.ref_s
            speed = prepared.speed

        if input_ids is None or input_lengths is None or ref_s is None or speed is None:
            raise ValueError(
                "input_ids, input_lengths, ref_s, and speed are required "
                "unless prepared is provided"
            )

        input_ids_np = _as_numpy(input_ids, np.int64)
        input_lengths_np = _as_numpy(input_lengths, np.int64)
        ref_s_np = _as_numpy(ref_s, np.float32)
        speed_np = _as_numpy(speed, np.float32)

        text_bucket = int(input_ids_np.shape[1])
        text_session = self._session_for(
            self.text_duration, text_bucket, ONNX_TEXT_DURATION_PREFIX
        )

        duration_float_np, duration_hidden_np, text_hidden_np = text_session.run(
            None,
            {
                "input_ids": input_ids_np,
                "input_lengths": input_lengths_np,
                "ref_s": ref_s_np,
                "speed": speed_np,
            },
        )

        duration_float = torch.from_numpy(duration_float_np)
        duration_hidden = torch.from_numpy(duration_hidden_np)
        text_hidden = torch.from_numpy(text_hidden_np)
        input_lengths_t = torch.from_numpy(input_lengths_np)

        frames = expand_token_features(
            duration_float,
            duration_hidden,
            text_hidden,
            input_lengths_t,
            self.frame_buckets,
        )

        acoustic_session = self._session_for(
            self.acoustic_vocoder,
            frames.frame_bucket,
            ONNX_ACOUSTIC_VOCODER_PREFIX,
        )
        audio_np = acoustic_session.run(
            None,
            {
                "asr": np.ascontiguousarray(frames.asr.numpy(), dtype=np.float32),
                "en": np.ascontiguousarray(frames.en.numpy(), dtype=np.float32),
                "ref_s": ref_s_np,
            },
        )[0]

        samples_per_frame = audio_np.shape[-1] // frames.frame_bucket
        out_len = frames.frame_lengths.max().item() * samples_per_frame
        audio_np = audio_np[..., :out_len]

        for b in range(audio_np.shape[0]):
            valid_samples = frames.frame_lengths[b].item() * samples_per_frame
            audio_np[b, ..., valid_samples:] = 0.0

        return KModel.Output(
            audio=torch.from_numpy(audio_np),
            pred_dur=frames.pred_dur,
            frame_lengths=frames.frame_lengths,
            duration_float=duration_float,
        )


class KONNXModel:
    """
    Lightweight ONNX counterpart to KModel.

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
        frame_buckets: Optional[Sequence[int]] = None,
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
            frame_buckets=frame_buckets,
        )
        self.text_buckets = self.backend.text_buckets
        self.frame_buckets = self.backend.frame_buckets

    def inference_backend(self):
        return self.backend

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        ref_s: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        prepared=None,
    ) -> "KModel.Output":
        return self.backend(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
            prepared=prepared,
        )
