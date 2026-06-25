from pathlib import Path
from typing import Any, Optional, Union

import torch
from loguru import logger
from torch.nn.utils import parametrize
from transformers import AlbertConfig

from .config import (MODEL_FILENAMES, load_config_data, resolve_model_path,
                     resolve_repo_id)
from .istftnet import (Decoder,
                       count_problematic_conv_transpose1d_for_tensorrt,
                       replace_conv_transpose1d_for_tensorrt)
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from .runtime import Synthesizer
from .types import FrameItem, KModelOutput


def remove_weight_norm_parametrizations(module: torch.nn.Module) -> None:
    for m in module.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)


class KokoroTextDuration(torch.nn.Module):
    """
    Exact-length text-duration module.

    Contract:
      input_ids: [1, T]
      ref_s:     [1, 256]
      speed:     [1]

    No padding, masks, packed sequences, or batch padding are used internally.
    Public batch-like inputs are sliced into exact single utterances by runtime.py.
    """

    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.bert = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder
        self.predictor = kmodel.predictor
        self.text_encoder = kmodel.text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ):
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"KokoroTextDuration expects input_ids [1,T], got {tuple(input_ids.shape)}"
            )
        if ref_s.dim() != 2 or ref_s.shape[0] != 1:
            raise ValueError(
                f"KokoroTextDuration expects ref_s [1,256], got {tuple(ref_s.shape)}"
            )

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
            0
        )

        bert_dur = self.bert(
            input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.int32),
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=positions,
        )
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        duration_style = ref_s[:, 128:]
        duration_hidden = self.predictor.text_encoder(d_en, duration_style)

        if not torch.jit.is_scripting():
            self.predictor.lstm.flatten_parameters()
        x, _ = self.predictor.lstm(duration_hidden)

        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(dim=-1)
        duration = duration / speed.reshape(-1, 1).to(duration.dtype)

        text_hidden = self.text_encoder(input_ids)
        return duration, duration_hidden, text_hidden


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


class KokoroDecodeGenerateWithHar(torch.nn.Module):
    """
    Tensor-only decoder/generator wrapper for TensorRT AOT.

    Host/PyTorch stages remain outside TensorRT:
      - text duration
      - frame expansion
      - F0/noise prediction
      - harmonic feature generation

    The inputs are not independently dynamic. They are tied by the synthesis-frame
    count:

        asr.shape[-1]       = S
        f0_curve.shape[-1]  = generator_frame_count(S)
        noise.shape[-1]     = generator_frame_count(S)
        har.shape[-1]       = harmonic_frame_count(S)

    Keep these assertions inside the exported graph so torch.export/TensorRT see
    the same shape contract that KokoroTRTBackend validates at runtime.
    """

    @staticmethod
    def _affine_relation(name: str, fn) -> tuple[int, int]:
        y1 = int(fn(1))
        y2 = int(fn(2))
        slope = y2 - y1
        intercept = y1 - slope

        if slope <= 0:
            raise ValueError(f"{name} must have positive slope, got {slope}")

        for frames in (1, 2, 3, 8):
            actual = int(fn(frames))
            expected = slope * frames + intercept
            if actual != expected:
                raise ValueError(
                    f"{name} is not affine in synthesis frame count: "
                    f"{name}({frames})={actual}, expected {expected} from "
                    f"{slope} * frames + {intercept}"
                )

        return slope, intercept

    @staticmethod
    def _apply_affine(dim, slope: int, intercept: int):
        value = dim if slope == 1 else dim * slope
        return value + intercept if intercept else value

    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.decoder = kmodel.decoder

        self.generator_frame_slope, self.generator_frame_intercept = (
            self._affine_relation(
                "generator_frame_count",
                kmodel.decoder.generator_input_frame_length,
            )
        )
        self.harmonic_frame_slope, self.harmonic_frame_intercept = (
            self._affine_relation(
                "harmonic_frame_count",
                lambda frames: kmodel.decoder.generator.output_frame_length(
                    kmodel.decoder.generator_input_frame_length(frames)
                ),
            )
        )

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        ref_s: torch.Tensor,
        har: torch.Tensor,
    ) -> torch.Tensor:
        torch._assert(asr.dim() == 3, "asr must have shape [1,C,S]")
        torch._assert(f0_curve.dim() == 2, "f0_curve must have shape [1,G]")
        torch._assert(noise.dim() == 2, "noise must have shape [1,G]")
        torch._assert(ref_s.dim() == 2, "ref_s must have shape [1,256]")
        torch._assert(har.dim() == 3, "har must have shape [1,C_har,H]")

        torch._assert(asr.shape[0] == 1, "TensorRT decoder batch size must be 1")
        torch._assert(f0_curve.shape[0] == 1, "f0_curve batch size must be 1")
        torch._assert(noise.shape[0] == 1, "noise batch size must be 1")
        torch._assert(ref_s.shape[0] == 1, "ref_s batch size must be 1")
        torch._assert(har.shape[0] == 1, "har batch size must be 1")
        torch._assert(ref_s.shape[1] == 256, "ref_s width must be 256")

        synthesis_frames = asr.shape[-1]
        expected_generator_frames = self._apply_affine(
            synthesis_frames,
            self.generator_frame_slope,
            self.generator_frame_intercept,
        )
        expected_har_frames = self._apply_affine(
            synthesis_frames,
            self.harmonic_frame_slope,
            self.harmonic_frame_intercept,
        )

        torch._assert(
            f0_curve.shape[-1] == expected_generator_frames,
            "f0_curve length must match generator_frame_count(asr frames)",
        )
        torch._assert(
            noise.shape[-1] == expected_generator_frames,
            "noise length must match generator_frame_count(asr frames)",
        )
        torch._assert(
            har.shape[-1] == expected_har_frames,
            "har length must match harmonic_frame_count(asr frames)",
        )

        return self.decoder.forward_with_har(
            asr,
            f0_curve,
            noise,
            ref_s[:, :128],
            har,
        )


class KokoroDecodeGenerateWithSourcePyramid(torch.nn.Module):
    """
    TensorRT AOT wrapper with an explicit harmonic/source pyramid.

    Host/PyTorch stages:
      - text duration
      - frame expansion
      - F0/noise prediction
      - harmonic feature generation
      - harmonic/source pyramid generation

    TensorRT stage:
      - Decoder.decode_features
      - Generator upsampling/resblocks/post/STFT inverse

    This avoids the TensorRT kMIN self-consistency failure caused by compiling
    the shape-fragile relationship:

        har -> strided noise_conv -> source_i
        asr -> decoder/generator upsample path -> source_i

    The source tensors are still validated against the same affine synthesis
    frame contract, but TensorRT no longer needs to infer the downsampling
    pyramid from the final harmonic tensor.
    """

    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.decoder = kmodel.decoder

        self.generator_frame_slope, self.generator_frame_intercept = (
            KokoroDecodeGenerateWithHar._affine_relation(
                "generator_frame_count",
                kmodel.decoder.generator_input_frame_length,
            )
        )

        self.source_channels = tuple(int(c) for c in kmodel.decoder.source_channels())
        if not self.source_channels:
            raise ValueError("Decoder/generator exposes no source-pyramid channels")

        source_lengths_at_one = kmodel.decoder.source_frame_lengths(1)
        if len(source_lengths_at_one) != len(self.source_channels):
            raise ValueError(
                "Decoder source channel/length metadata mismatch: "
                f"{len(self.source_channels)} channel entries, "
                f"{len(source_lengths_at_one)} length entries"
            )

        self.source_frame_relations = tuple(
            KokoroDecodeGenerateWithHar._affine_relation(
                f"source_{i}_frame_count",
                lambda frames, i=i: kmodel.decoder.source_frame_lengths(frames)[i],
            )
            for i in range(len(self.source_channels))
        )

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise: torch.Tensor,
        ref_s: torch.Tensor,
        *source_pyramid: torch.Tensor,
    ) -> torch.Tensor:
        if len(source_pyramid) != len(self.source_channels):
            raise ValueError(
                "source_pyramid input count mismatch: "
                f"got {len(source_pyramid)}, expected {len(self.source_channels)}"
            )

        torch._assert(asr.dim() == 3, "asr must have shape [1,C,S]")
        torch._assert(f0_curve.dim() == 2, "f0_curve must have shape [1,G]")
        torch._assert(noise.dim() == 2, "noise must have shape [1,G]")
        torch._assert(ref_s.dim() == 2, "ref_s must have shape [1,256]")

        torch._assert(asr.shape[0] == 1, "TensorRT decoder batch size must be 1")
        torch._assert(f0_curve.shape[0] == 1, "f0_curve batch size must be 1")
        torch._assert(noise.shape[0] == 1, "noise batch size must be 1")
        torch._assert(ref_s.shape[0] == 1, "ref_s batch size must be 1")
        torch._assert(ref_s.shape[1] == 256, "ref_s width must be 256")

        synthesis_frames = asr.shape[-1]
        expected_generator_frames = KokoroDecodeGenerateWithHar._apply_affine(
            synthesis_frames,
            self.generator_frame_slope,
            self.generator_frame_intercept,
        )

        torch._assert(
            f0_curve.shape[-1] == expected_generator_frames,
            "f0_curve length must match generator_frame_count(asr frames)",
        )
        torch._assert(
            noise.shape[-1] == expected_generator_frames,
            "noise length must match generator_frame_count(asr frames)",
        )

        for i, source in enumerate(source_pyramid):
            expected_source_frames = KokoroDecodeGenerateWithHar._apply_affine(
                synthesis_frames,
                self.source_frame_relations[i][0],
                self.source_frame_relations[i][1],
            )

            torch._assert(source.dim() == 3, "source tensor must have shape [1,C,T]")
            torch._assert(source.shape[0] == 1, "source tensor batch size must be 1")
            torch._assert(
                source.shape[1] == self.source_channels[i],
                "source tensor channel count mismatch",
            )
            torch._assert(
                source.shape[-1] == expected_source_frames,
                "source tensor length must match its generator layer frame count",
            )

        return self.decoder.forward_with_source_pyramid(
            asr,
            f0_curve,
            noise,
            ref_s[:, :128],
            source_pyramid,
        )


class KokoroGenerateWithSourcePyramid(torch.nn.Module):
    """
    TensorRT AOT wrapper for only the ISTFTNet generator.

    Host/PyTorch stages:
      - text duration
      - frame expansion
      - ProsodyPredictor.F0Ntrain
      - Decoder.decode_features
      - harmonic feature generation
      - harmonic/source pyramid generation

    TensorRT stage:
      - Generator.forward_with_source_pyramid

    This is intentionally narrower than KokoroDecodeGenerateWithSourcePyramid.
    The previous decoder+generator TensorRT graph still required TensorRT to
    prove this dynamic relationship:

        asr frames -> Decoder.decode_features upsample -> generator frames
        generator frames -> Generator upsample path -> source_i frames

    That relationship is valid in PyTorch, but TensorRT's shape machine can
    reject the kMIN profile with elementwise-add mismatches such as 20 != 40.
    Passing the decoded generator input directly makes the TensorRT graph's
    dynamic contract simple:

        x.shape[-1]        = G
        source_i.shape[-1] = affine_i(G)
    """

    def __init__(self, kmodel: "KModel"):
        super().__init__()
        self.generator = kmodel.decoder.generator

        if not self.generator.ups:
            raise ValueError("Generator exposes no upsampling layers")

        self.input_channels = int(self.generator.ups[0].in_channels)

        self.source_channels = tuple(int(c) for c in self.generator.source_channels())
        if not self.source_channels:
            raise ValueError("Generator exposes no source-pyramid channels")

        source_lengths_at_one = self.generator.source_frame_lengths(1)
        if len(source_lengths_at_one) != len(self.source_channels):
            raise ValueError(
                "Generator source channel/length metadata mismatch: "
                f"{len(self.source_channels)} channel entries, "
                f"{len(source_lengths_at_one)} length entries"
            )

        self.source_frame_relations = tuple(
            KokoroDecodeGenerateWithHar._affine_relation(
                f"source_{i}_frame_count_from_generator_frames",
                lambda generator_frames, i=i: self.generator.source_frame_lengths(
                    generator_frames
                )[i],
            )
            for i in range(len(self.source_channels))
        )

    def forward(
        self,
        x: torch.Tensor,
        ref_s: torch.Tensor,
        *source_pyramid: torch.Tensor,
    ) -> torch.Tensor:
        if len(source_pyramid) != len(self.source_channels):
            raise ValueError(
                "source_pyramid input count mismatch: "
                f"got {len(source_pyramid)}, expected {len(self.source_channels)}"
            )

        torch._assert(x.dim() == 3, "x must have shape [1,C,G]")
        torch._assert(ref_s.dim() == 2, "ref_s must have shape [1,256]")

        torch._assert(x.shape[0] == 1, "TensorRT generator batch size must be 1")
        torch._assert(ref_s.shape[0] == 1, "ref_s batch size must be 1")
        torch._assert(ref_s.shape[1] == 256, "ref_s width must be 256")
        torch._assert(
            x.shape[1] == self.input_channels,
            "generator input channel count mismatch",
        )

        generator_frames = x.shape[-1]

        for i, source in enumerate(source_pyramid):
            expected_source_frames = KokoroDecodeGenerateWithHar._apply_affine(
                generator_frames,
                self.source_frame_relations[i][0],
                self.source_frame_relations[i][1],
            )

            torch._assert(source.dim() == 3, "source tensor must have shape [1,C,T]")
            torch._assert(source.shape[0] == 1, "source tensor batch size must be 1")
            torch._assert(
                source.shape[1] == self.source_channels[i],
                "source tensor channel count mismatch",
            )
            torch._assert(
                source.shape[-1] == expected_source_frames,
                "source tensor length must match its generator layer frame count",
            )

        return self.generator.forward_with_source_pyramid(
            x,
            ref_s[:, :128],
            source_pyramid,
        )


class KokoroInferenceBackend:
    """
    PyTorch backend implementation.

    This class implements the shared backend interface consumed by Synthesizer:
      - text_duration(input_ids, ref_s, speed)
      - render(frame_item, ref_s)
    """

    def __init__(self, kmodel: "KModel"):
        self.kmodel = kmodel.eval()
        self.device = self.kmodel.device
        self.text_duration_module = KokoroTextDuration(kmodel).eval()
        self.acoustic_vocoder = KokoroAcousticVocoder(kmodel).eval()
        self.preferred_ref_device = self.device
        self.preferred_ref_dtype = torch.float32
        self.synthesizer = Synthesizer(self)

    def text_duration(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ):
        return self.text_duration_module(input_ids, ref_s, speed)

    def render(self, frame_item: FrameItem, ref_s: torch.Tensor) -> torch.Tensor:
        asr = frame_item.asr.unsqueeze(0)
        en = frame_item.en.unsqueeze(0)

        f0, n = self.acoustic_vocoder.predict_f0n(en, ref_s)
        har = self.kmodel.compute_harmonic_features(f0)
        return self.acoustic_vocoder.forward_with_f0n(asr, f0, n, ref_s, har)

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


class KModel(torch.nn.Module):
    Output = KModelOutput
    MODEL_NAMES = MODEL_FILENAMES

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[dict[str, Any], str, Path, None] = None,
        model: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.repo_id = resolve_repo_id(repo_id)
        self.config_data = load_config_data(self.repo_id, config)

        self.vocab: Optional[dict[str, int]] = self.config_data.get("vocab")
        self.bert = CustomAlbert(
            AlbertConfig(
                vocab_size=self.config_data["n_token"], **self.config_data["plbert"]
            )
        )
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size,
            self.config_data["hidden_dim"],
        )
        self.context_length: int = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
            style_dim=self.config_data["style_dim"],
            d_hid=self.config_data["hidden_dim"],
            nlayers=self.config_data["n_layer"],
            max_dur=self.config_data["max_dur"],
            dropout=self.config_data["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=self.config_data["hidden_dim"],
            kernel_size=self.config_data["text_encoder_kernel_size"],
            depth=self.config_data["n_layer"],
            n_symbols=self.config_data["n_token"],
        )
        self.decoder = Decoder(
            dim_in=self.config_data["hidden_dim"],
            style_dim=self.config_data["style_dim"],
            resblock_kernel_sizes=self.config_data["istftnet"]["resblock_kernel_sizes"],
            upsample_rates=self.config_data["istftnet"]["upsample_rates"],
            upsample_initial_channel=self.config_data["istftnet"][
                "upsample_initial_channel"
            ],
            resblock_dilation_sizes=self.config_data["istftnet"][
                "resblock_dilation_sizes"
            ],
            upsample_kernel_sizes=self.config_data["istftnet"]["upsample_kernel_sizes"],
            gen_istft_n_fft=self.config_data["istftnet"]["gen_istft_n_fft"],
            gen_istft_hop_size=self.config_data["istftnet"]["gen_istft_hop_size"],
        )

        self.model_path = resolve_model_path(self.repo_id, model)

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        for key, state_dict in checkpoint.items():
            assert hasattr(self, key), key
            self._load_submodule_state(key, state_dict)

        self.remove_weight_norm()
        self._inference_backend: Optional[KokoroInferenceBackend] = None

    def _load_submodule_state(self, key: str, state_dict: dict[str, torch.Tensor]):
        module = getattr(self, key)
        try:
            module.load_state_dict(state_dict)
            return
        except Exception as first_error:
            if state_dict and all(k.startswith("module.") for k in state_dict):
                stripped = {k[7:]: v for k, v in state_dict.items()}
                try:
                    module.load_state_dict(stripped)
                    return
                except Exception:
                    state_dict = stripped

            logger.debug(
                f"Strict load failed for {key}; retrying strict=False: {first_error}"
            )
            incompatible = module.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.debug(f"{key} missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.debug(f"{key} unexpected keys: {incompatible.unexpected_keys}")

    @property
    def device(self):
        return next(self.parameters()).device

    def remove_weight_norm(self):
        remove_weight_norm_parametrizations(self)

    def prepare_for_export(self):
        self.eval()
        self.remove_weight_norm()
        return self

    def prepare_for_tensorrt_export(self):
        self.prepare_for_export()

        replacements = replace_conv_transpose1d_for_tensorrt(self.decoder)
        remaining = count_problematic_conv_transpose1d_for_tensorrt(self.decoder)

        if remaining:
            raise RuntimeError(
                "TensorRT export preparation failed: "
                f"{remaining} ConvTranspose1d module(s) remain in the decoder."
            )

        logger.debug(
            "Prepared decoder for TensorRT: replaced {} ConvTranspose1d module(s) "
            "with exact phase-decomposed Conv1d equivalents.",
            replacements,
        )

        return self

    def compute_harmonic_features(self, f0: torch.Tensor):
        return self.decoder.generator.compute_harmonic_features(f0)

    def text_duration_module(self):
        return KokoroTextDuration(self).eval()

    def acoustic_vocoder_module(self):
        return KokoroAcousticVocoder(self).eval()

    def decode_generate_with_har_module(self):
        return KokoroDecodeGenerateWithHar(self).eval()

    def decode_generate_with_source_pyramid_module(self):
        return KokoroGenerateWithSourcePyramid(self).eval()

    def generate_with_source_pyramid_module(self):
        return KokoroGenerateWithSourcePyramid(self).eval()

    def inference_backend(self):
        if self._inference_backend is None:
            self._inference_backend = KokoroInferenceBackend(self)
        return self._inference_backend

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: Optional[torch.Tensor],
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ) -> KModelOutput:
        return self.inference_backend()(
            input_ids=input_ids,
            input_lengths=input_lengths,
            ref_s=ref_s,
            speed=speed,
        )
