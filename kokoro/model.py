import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from torch.nn.utils import parametrize
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder, run_length_aware_lstm

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"

ONNX_TEXT_DURATION_PREFIX = "text_duration"
ONNX_ACOUSTIC_VOCODER_PREFIX = "acoustic_vocoder"

END_SILENCE_FRAMES = 5
KEEP_EOS_FRAMES = 1


def resolve_repo_id(repo_id: Optional[str]) -> str:
    if repo_id is None:
        repo_id = DEFAULT_REPO_ID
        print(
            f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning."
        )
    return repo_id


def load_config_data(
    repo_id: str, config: Union[dict[str, Any], str, Path, None] = None
) -> dict[str, Any]:
    if isinstance(config, dict):
        return config

    config_path = config
    if not config_path:
        logger.debug("No config provided, downloading from HF")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    with open(config_path, "r", encoding="utf-8") as r:
        return json.load(r)


def onnx_export_path(output_dir: Union[str, Path], prefix: str) -> Path:
    return Path(output_dir) / f"{prefix}.onnx"


def remove_weight_norm_parametrizations(module: torch.nn.Module) -> None:
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


@dataclass
class FrameItem:
    asr: torch.Tensor
    en: torch.Tensor
    pred_dur: torch.Tensor
    synthesis_frame_length: int
    return_frame_length: int


@dataclass
class FrameExpansion:
    items: list[FrameItem]


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
class InferenceBatch:
    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    ref_s: torch.Tensor
    speed: torch.Tensor
    graphemes: list[Optional[str]]
    phonemes: list[Optional[str]]


def normalize_inference_inputs(
    input_ids: Optional[torch.Tensor] = None,
    input_lengths: Optional[torch.Tensor] = None,
    ref_s: Optional[torch.Tensor] = None,
    speed: Optional[torch.Tensor] = None,
    prepared=None,
    device: Optional[torch.device] = None,
) -> InferenceBatch:
    if prepared is not None:
        input_ids = prepared.input_ids
        input_lengths = torch.tensor([int(prepared.input_length)], dtype=torch.long)
        ref_s = prepared.ref_s
        speed = torch.tensor([float(prepared.speed)], dtype=torch.float32)
        graphemes = [getattr(prepared, "graphemes", None)]
        phonemes = [getattr(prepared, "phonemes", None)]
    else:
        graphemes = []
        phonemes = []

    if input_ids is None or ref_s is None or speed is None:
        raise ValueError(
            "input_ids, ref_s, and speed are required unless prepared is provided"
        )

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    else:
        input_ids = input_ids.to(dtype=torch.long)

    if input_lengths is None:
        if input_ids.dim() == 1:
            input_lengths = torch.tensor([input_ids.shape[0]], dtype=torch.long)
        elif input_ids.dim() == 2:
            input_lengths = torch.full(
                (input_ids.shape[0],), input_ids.shape[1], dtype=torch.long
            )
        else:
            raise ValueError(
                f"Expected input_ids [T] or [B,T], got {tuple(input_ids.shape)}"
            )
    elif not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    else:
        input_lengths = input_lengths.to(dtype=torch.long)

    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.as_tensor(ref_s, dtype=torch.float32)
    else:
        ref_s = ref_s.to(dtype=torch.float32)

    if not isinstance(speed, torch.Tensor):
        speed = torch.as_tensor(speed, dtype=torch.float32)
    else:
        speed = speed.to(dtype=torch.float32)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_lengths.dim() == 0:
        input_lengths = input_lengths.unsqueeze(0)
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    if speed.dim() == 0:
        speed = speed.unsqueeze(0)

    batch_size = input_ids.shape[0]
    if not graphemes:
        graphemes = [None] * batch_size
        phonemes = [None] * batch_size

    if device is not None:
        input_ids = input_ids.to(device)
        input_lengths = input_lengths.to(device)
        ref_s = ref_s.to(device)
        speed = speed.to(device)

    return InferenceBatch(
        input_ids=input_ids,
        input_lengths=input_lengths,
        ref_s=ref_s,
        speed=speed,
        graphemes=graphemes,
        phonemes=phonemes,
    )


def expand_token_features(
    duration_float: torch.Tensor,
    duration_hidden: torch.Tensor,
    text_hidden: torch.Tensor,
    input_lengths: torch.Tensor,
    end_silence_frames: int = END_SILENCE_FRAMES,
    keep_eos_frames: int = KEEP_EOS_FRAMES,
) -> FrameExpansion:
    device = duration_float.device
    batch = duration_float.shape[0]
    items: list[FrameItem] = []

    for b in range(batch):
        n = int(input_lengths[b].item())
        if n <= 0:
            raise ValueError("input_lengths must be positive")

        d = torch.round(duration_float[b, :n]).clamp(min=1).long()

        if end_silence_frames > 0:
            d[-1] += end_silence_frames

        idx = torch.repeat_interleave(torch.arange(n, device=device), d)
        synthesis_frame_length = int(idx.numel())

        asr = text_hidden[b, idx, :].transpose(0, 1).contiguous()
        en = duration_hidden[b, idx, :].transpose(0, 1).contiguous()

        eos_frames = int(d[-1].item())
        trim_frames = max(eos_frames - keep_eos_frames, 0)
        return_frame_length = max(1, synthesis_frame_length - trim_frames)

        items.append(
            FrameItem(
                asr=asr,
                en=en,
                pred_dur=d,
                synthesis_frame_length=synthesis_frame_length,
                return_frame_length=return_frame_length,
            )
        )

    return FrameExpansion(items=items)


class KokoroTextDuration(torch.nn.Module):
    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.bert = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder
        self.predictor = kmodel.predictor
        self.text_encoder = kmodel.text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )
        text_mask = positions >= input_lengths.unsqueeze(1)
        valid = (~text_mask).to(dtype=ref_s.dtype)

        bert_dur = self.bert(
            input_ids,
            attention_mask=valid.to(torch.int32),
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=positions,
        )
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        duration_style = ref_s[:, 128:]
        duration_hidden = self.predictor.text_encoder(
            d_en, duration_style, input_lengths, text_mask
        )

        x = run_length_aware_lstm(
            self.predictor.lstm,
            duration_hidden,
            input_lengths,
            total_length=duration_hidden.shape[1],
        )
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(dim=-1)
        duration = duration / speed.reshape(-1, 1).to(duration.dtype)
        duration = duration * valid

        text_hidden = self.text_encoder(input_ids, input_lengths, text_mask).transpose(
            -1, -2
        )
        text_hidden = text_hidden * valid.unsqueeze(-1)
        duration_hidden = duration_hidden * valid.unsqueeze(-1)

        return duration, duration_hidden, text_hidden

    def export_onnx(
        self,
        path: str,
        example_text_length: int = 64,
        opset: int = 18,
    ):
        self.eval()
        device = next(self.parameters()).device
        args = (
            torch.zeros((1, example_text_length), dtype=torch.long, device=device),
            torch.full((1,), example_text_length, dtype=torch.long, device=device),
            torch.zeros((1, 256), dtype=torch.float32, device=device),
            torch.ones((1,), dtype=torch.float32, device=device),
        )
        text_len = torch.export.Dim("text_len", min=1)
        torch.onnx.export(
            self,
            args,
            path,
            input_names=["input_ids", "input_lengths", "ref_s", "speed"],
            output_names=["duration_float", "duration_hidden", "text_hidden"],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes={
                "input_ids": {1: text_len},
                "input_lengths": {},
                "ref_s": {},
                "speed": {},
            },
            dynamic_axes={
                "input_ids": {1: "text_len"},
                "duration_float": {1: "text_len"},
                "duration_hidden": {1: "text_len"},
                "text_hidden": {1: "text_len"},
            },
        )


class KokoroAcousticVocoder(torch.nn.Module):
    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.predictor = kmodel.predictor
        self.decoder = kmodel.decoder
        self.asr_channels = kmodel.bert_encoder.out_features
        self.en_channels = kmodel.predictor.shared.input_size

    def predict_f0n(self, en: torch.Tensor, ref_s: torch.Tensor):
        return self.predictor.F0Ntrain(en, ref_s[:, 128:])

    def forward_with_f0n(
        self,
        asr: torch.Tensor,
        f0: torch.Tensor,
        n: torch.Tensor,
        ref_s: torch.Tensor,
        har: torch.Tensor,
    ):
        return self.decoder.forward_with_har(asr, f0, n, ref_s[:, :128], har)

    def forward(
        self,
        asr: torch.Tensor,
        en: torch.Tensor,
        ref_s: torch.Tensor,
    ):
        f0, n = self.predict_f0n(en, ref_s)
        return self.decoder(asr, f0, n, ref_s[:, :128])

    def export_onnx(
        self,
        path: str,
        example_frame_length: int = 128,
        opset: int = 18,
    ):
        self.eval()
        device = next(self.parameters()).device
        args = (
            torch.zeros(
                (1, self.asr_channels, example_frame_length),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (1, self.en_channels, example_frame_length),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros((1, 256), dtype=torch.float32, device=device),
        )
        frame_len = torch.export.Dim("frame_len", min=1)
        torch.onnx.export(
            self,
            args,
            path,
            input_names=["asr", "en", "ref_s"],
            output_names=["waveform"],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes={
                "asr": {2: frame_len},
                "en": {2: frame_len},
                "ref_s": {},
            },
            dynamic_axes={
                "asr": {2: "frame_len"},
                "en": {2: "frame_len"},
                "waveform": {2: "sample_len"},
            },
        )


class KokoroInferenceBackend:
    def __init__(self, kmodel: "KModel"):
        self.kmodel = kmodel.eval()
        self.text_duration = KokoroTextDuration(kmodel).eval()
        self.acoustic_vocoder = KokoroAcousticVocoder(kmodel).eval()

    @torch.no_grad()
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
            device=self.kmodel.device,
        )

        duration_float, duration_hidden, text_hidden = self.text_duration(
            batch.input_ids,
            batch.input_lengths,
            batch.ref_s,
            batch.speed,
        )
        frames = expand_token_features(
            duration_float,
            duration_hidden,
            text_hidden,
            batch.input_lengths,
            end_silence_frames=END_SILENCE_FRAMES,
            keep_eos_frames=KEEP_EOS_FRAMES,
        )

        utterances: list[UtteranceOutput] = []

        for b, item in enumerate(frames.items):
            asr = item.asr.unsqueeze(0)
            en = item.en.unsqueeze(0)
            ref = batch.ref_s[b : b + 1]

            f0, n = self.acoustic_vocoder.predict_f0n(en, ref)
            har = self.kmodel.compute_harmonic_features(f0)
            audio = self.acoustic_vocoder.forward_with_f0n(asr, f0, n, ref, har)

            samples_per_frame = audio.shape[-1] // item.synthesis_frame_length
            sample_length = item.return_frame_length * samples_per_frame
            audio = audio[..., :sample_length].reshape(-1).contiguous()

            text_len = int(batch.input_lengths[b].item())
            utterances.append(
                UtteranceOutput(
                    audio=audio,
                    pred_dur=item.pred_dur,
                    duration_float=duration_float[b, :text_len].contiguous(),
                    synthesis_frame_length=item.synthesis_frame_length,
                    return_frame_length=item.return_frame_length,
                    sample_length=sample_length,
                    graphemes=batch.graphemes[b],
                    phonemes=batch.phonemes[b],
                )
            )

        return KModelOutput(utterances=utterances)


class KModel(torch.nn.Module):
    Output = KModelOutput

    MODEL_NAMES: dict[str, str] = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[dict[str, Any], str, Path, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        self.repo_id = resolve_repo_id(repo_id)
        config_data = load_config_data(self.repo_id, config)

        self.vocab: Optional[dict[str, int]] = config_data.get("vocab")
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=config_data["n_token"], **config_data["plbert"])
        )
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size, config_data["hidden_dim"]
        )
        self.context_length: int = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
            style_dim=config_data["style_dim"],
            d_hid=config_data["hidden_dim"],
            nlayers=config_data["n_layer"],
            max_dur=config_data["max_dur"],
            dropout=config_data["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config_data["hidden_dim"],
            kernel_size=config_data["text_encoder_kernel_size"],
            depth=config_data["n_layer"],
            n_symbols=config_data["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config_data["hidden_dim"],
            style_dim=config_data["style_dim"],
            dim_out=config_data["n_mels"],
            disable_complex=disable_complex,
            **config_data["istftnet"],
        )

        if not model:
            model = hf_hub_download(
                repo_id=self.repo_id, filename=KModel.MODEL_NAMES[self.repo_id]
            )

        for key, state_dict in torch.load(
            model, map_location="cpu", weights_only=True
        ).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except Exception:
                logger.debug(
                    f"Did not directly load {key}; retrying with stripped module prefix"
                )
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

        self.remove_weight_norm()

    @property
    def device(self):
        return next(self.parameters()).device

    def remove_weight_norm(self):
        remove_weight_norm_parametrizations(self)

    def prepare_for_export(self):
        self.eval()
        self.remove_weight_norm()
        return self

    def compute_harmonic_features(self, f0: torch.Tensor):
        return self.decoder.generator.compute_harmonic_features(f0)

    def text_duration_module(self):
        return KokoroTextDuration(self).eval()

    def acoustic_vocoder_module(self):
        return KokoroAcousticVocoder(self).eval()

    def inference_backend(self):
        return KokoroInferenceBackend(self)

    def export_text_duration_onnx(
        self,
        path: str,
        example_text_length: int = 64,
        opset: int = 18,
    ):
        self.prepare_for_export()
        return self.text_duration_module().export_onnx(
            path,
            example_text_length=example_text_length,
            opset=opset,
        )

    def export_acoustic_vocoder_onnx(
        self,
        path: str,
        example_frame_length: int = 128,
        opset: int = 18,
    ):
        self.prepare_for_export()
        return self.acoustic_vocoder_module().export_onnx(
            path,
            example_frame_length=example_frame_length,
            opset=opset,
        )

    def export_onnx(
        self,
        output_dir: Union[str, Path],
        opset: int = 18,
        example_text_length: int = 64,
        example_frame_length: int = 128,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.prepare_for_export()
        text_duration = self.text_duration_module()
        acoustic_vocoder = self.acoustic_vocoder_module()

        text_path = onnx_export_path(output_dir, ONNX_TEXT_DURATION_PREFIX)
        acoustic_path = onnx_export_path(output_dir, ONNX_ACOUSTIC_VOCODER_PREFIX)

        text_duration.export_onnx(
            str(text_path),
            example_text_length=example_text_length,
            opset=opset,
        )
        acoustic_vocoder.export_onnx(
            str(acoustic_path),
            example_frame_length=example_frame_length,
            opset=opset,
        )

        return {
            ONNX_TEXT_DURATION_PREFIX: text_path,
            ONNX_ACOUSTIC_VOCODER_PREFIX: acoustic_path,
        }

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ) -> KModelOutput:
        return self.inference_backend()(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
        )
