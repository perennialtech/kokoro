from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from torch.nn.utils import parametrize
from transformers import AlbertConfig
from typing import Any, Optional, Sequence, Union
import json
import torch


def remove_weight_norm_parametrizations(module: torch.nn.Module) -> None:
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


def _bucket_for(length: int, buckets: Sequence[int]) -> int:
    for b in sorted(buckets):
        if length <= b:
            return b
    raise ValueError(f"Length {length} exceeds largest bucket {max(buckets)}")


@dataclass
class FrameExpansion:
    asr: torch.Tensor
    en: torch.Tensor
    frame_lengths: torch.Tensor
    pred_dur: torch.Tensor
    frame_bucket: int


def expand_token_features(
    duration_float: torch.Tensor,
    duration_hidden: torch.Tensor,
    text_hidden: torch.Tensor,
    input_lengths: torch.Tensor,
    frame_buckets: Sequence[int] = (128, 256, 512, 1024, 2048, 4096),
) -> FrameExpansion:
    """
    Host-side duration rounding and token-to-frame expansion.

    duration_float: [B,T]
    duration_hidden: [B,T,H_en]
    text_hidden: [B,T,H_asr]
    input_lengths: [B]
    """
    device = duration_float.device
    batch, text_bucket = duration_float.shape
    pred_dur = torch.zeros((batch, text_bucket), dtype=torch.long, device=device)
    totals = []

    for b in range(batch):
        n = int(input_lengths[b].item())
        d = torch.round(duration_float[b, :n]).clamp(min=1).long()
        pred_dur[b, :n] = d
        totals.append(int(d.sum().item()))

    frame_bucket = _bucket_for(max(totals), frame_buckets)
    asr_channels = text_hidden.shape[-1]
    en_channels = duration_hidden.shape[-1]
    asr = text_hidden.new_zeros((batch, asr_channels, frame_bucket))
    en = duration_hidden.new_zeros((batch, en_channels, frame_bucket))

    for b in range(batch):
        n = int(input_lengths[b].item())
        idx = torch.repeat_interleave(torch.arange(n, device=device), pred_dur[b, :n])
        total = idx.numel()
        asr[b, :, :total] = text_hidden[b, idx, :].transpose(0, 1)
        en[b, :, :total] = duration_hidden[b, idx, :].transpose(0, 1)

    return FrameExpansion(
        asr=asr,
        en=en,
        frame_lengths=torch.tensor(totals, dtype=torch.long, device=device),
        pred_dur=pred_dur,
        frame_bucket=frame_bucket,
    )


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

        bert_dur = self.bert(input_ids, attention_mask=valid.to(torch.int32))
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        duration_style = ref_s[:, 128:]
        duration_hidden = self.predictor.text_encoder(
            d_en, duration_style, input_lengths, text_mask
        )

        self.predictor.lstm.flatten_parameters()
        x, _ = self.predictor.lstm(duration_hidden)
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
        self, path: str, batch_size: int = 1, text_bucket: int = 512, opset: int = 18
    ):
        self.eval()
        device = next(self.parameters()).device
        args = (
            torch.zeros((batch_size, text_bucket), dtype=torch.long, device=device),
            torch.full((batch_size,), text_bucket, dtype=torch.long, device=device),
            torch.zeros((batch_size, 256), dtype=torch.float32, device=device),
            torch.ones((batch_size,), dtype=torch.float32, device=device),
        )
        torch.onnx.export(
            self,
            args,
            path,
            input_names=["input_ids", "input_lengths", "ref_s", "speed"],
            output_names=["duration_float", "duration_hidden", "text_hidden"],
            opset_version=opset,
            dynamo=True,
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
        har: torch.Tensor,
    ):
        f0, n = self.predict_f0n(en, ref_s)
        return self.forward_with_f0n(asr, f0, n, ref_s, har)

    def export_onnx(
        self,
        path: str,
        batch_size: int = 1,
        frame_bucket: int = 512,
        har_frames: Optional[int] = None,
        opset: int = 18,
    ):
        self.eval()
        device = next(self.parameters()).device
        har_frames = har_frames or frame_bucket
        n_fft_plus_2 = self.decoder.generator.post_n_fft + 2
        args = (
            torch.zeros(
                (batch_size, self.asr_channels, frame_bucket),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (batch_size, self.en_channels, frame_bucket),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros((batch_size, 256), dtype=torch.float32, device=device),
            torch.zeros(
                (batch_size, n_fft_plus_2, har_frames),
                dtype=torch.float32,
                device=device,
            ),
        )
        torch.onnx.export(
            self,
            args,
            path,
            input_names=["asr", "en", "ref_s", "har"],
            output_names=["waveform"],
            opset_version=opset,
            dynamo=True,
        )


class KokoroInferenceBackend:
    def __init__(
        self,
        kmodel: "KModel",
        frame_buckets: Sequence[int] = (128, 256, 512, 1024, 2048, 4096),
    ):
        self.kmodel = kmodel.eval()
        self.text_duration = KokoroTextDuration(kmodel).eval()
        self.acoustic_vocoder = KokoroAcousticVocoder(kmodel).eval()
        self.frame_buckets = tuple(frame_buckets)

    @torch.no_grad()
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

        device = self.kmodel.device
        input_ids = input_ids.to(device)
        input_lengths = input_lengths.to(device)
        ref_s = ref_s.to(device)
        speed = speed.to(device)

        duration_float, duration_hidden, text_hidden = self.text_duration(
            input_ids, input_lengths, ref_s, speed
        )
        frames = expand_token_features(
            duration_float,
            duration_hidden,
            text_hidden,
            input_lengths,
            self.frame_buckets,
        )

        f0, n = self.acoustic_vocoder.predict_f0n(frames.en, ref_s)
        har = self.kmodel.compute_harmonic_features(f0)
        audio = self.acoustic_vocoder.forward_with_f0n(frames.asr, f0, n, ref_s, har)

        return KModel.Output(
            audio=audio,
            pred_dur=frames.pred_dur,
            frame_lengths=frames.frame_lengths,
            duration_float=duration_float,
        )


class KModel(torch.nn.Module):
    MODEL_NAMES: dict[str, str] = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[dict[str, Any], str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        if repo_id is None:
            repo_id = "hexgrad/Kokoro-82M"
            print(
                f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning."
            )
        self.repo_id: str = repo_id

        if isinstance(config, dict):
            config_data = config
        else:
            config_path = config
            if not config_path:
                logger.debug("No config provided, downloading from HF")
                config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as r:
                config_data: dict[str, Any] = json.load(r)

        self.vocab: dict[str, int] = config_data["vocab"]
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
                repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id]
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

    def inference_backend(
        self, frame_buckets: Sequence[int] = (128, 256, 512, 1024, 2048, 4096)
    ):
        return KokoroInferenceBackend(self, frame_buckets=frame_buckets)

    def export_text_duration_onnx(
        self, path: str, batch_size: int = 1, text_bucket: int = 512, opset: int = 18
    ):
        self.prepare_for_export()
        return self.text_duration_module().export_onnx(
            path, batch_size=batch_size, text_bucket=text_bucket, opset=opset
        )

    def export_acoustic_vocoder_onnx(
        self,
        path: str,
        batch_size: int = 1,
        frame_bucket: int = 512,
        har_frames: Optional[int] = None,
        opset: int = 18,
    ):
        self.prepare_for_export()
        return self.acoustic_vocoder_module().export_onnx(
            path,
            batch_size=batch_size,
            frame_bucket=frame_bucket,
            har_frames=har_frames,
            opset=opset,
        )

    @dataclass
    class Output:
        audio: torch.Tensor
        pred_dur: Optional[torch.Tensor] = None
        frame_lengths: Optional[torch.Tensor] = None
        duration_float: Optional[torch.Tensor] = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ) -> "KModel.Output":
        return self.inference_backend()(
            input_ids=input_ids, input_lengths=input_lengths, ref_s=ref_s, speed=speed
        )
