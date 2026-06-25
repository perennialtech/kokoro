from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from loguru import logger

from .config import load_trt_metadata
from .model import KokoroAcousticVocoder, KokoroTextDuration
from .runtime import Synthesizer
from .types import FrameItem, KModelOutput

TRT_ENGINE_FILENAME = "generator_with_source_pyramid.pt2"


@dataclass(frozen=True)
class TensorRTDynamicShapeProfile:
    min_frames: int
    opt_frames: int
    max_frames: int

    def validate(self) -> None:
        if self.min_frames < 1:
            raise ValueError("min_frames must be positive")
        if self.opt_frames < self.min_frames:
            raise ValueError("opt_frames must be >= min_frames")
        if self.max_frames < self.opt_frames:
            raise ValueError("max_frames must be >= opt_frames")


def generator_frame_count(kmodel, synthesis_frames: int) -> int:
    return int(kmodel.decoder.generator_input_frame_length(int(synthesis_frames)))


def harmonic_frame_count(kmodel, synthesis_frames: int) -> int:
    generator_frames = generator_frame_count(kmodel, int(synthesis_frames))
    return int(kmodel.decoder.generator.output_frame_length(generator_frames))


def source_frame_counts(kmodel, synthesis_frames: int) -> list[int]:
    return [
        int(length)
        for length in kmodel.decoder.source_frame_lengths(int(synthesis_frames))
    ]


def generator_input_channels(kmodel) -> int:
    if not kmodel.decoder.generator.ups:
        raise ValueError("Generator exposes no upsampling layers")
    return int(kmodel.decoder.generator.ups[0].in_channels)


def generator_profile_shapes(
    kmodel,
    profile: TensorRTDynamicShapeProfile,
) -> dict[str, dict[str, tuple[int, ...]]]:
    profile.validate()

    input_channels = generator_input_channels(kmodel)

    def f(synthesis_frames: int) -> dict[str, tuple[int, ...]]:
        generator_frames = generator_frame_count(kmodel, synthesis_frames)
        result: dict[str, tuple[int, ...]] = {
            "x": (1, input_channels, generator_frames),
            "ref_s": (1, 256),
        }

        for i, (channels, source_frames) in enumerate(
            zip(
                kmodel.decoder.source_channels(),
                source_frame_counts(kmodel, synthesis_frames),
            )
        ):
            result[f"source_{i}"] = (1, int(channels), int(source_frames))

        return result

    return {
        "min": f(profile.min_frames),
        "opt": f(profile.opt_frames),
        "max": f(profile.max_frames),
    }


class KokoroTRTBackend:
    """
    Explicit TensorRT backend using the shared exact synthesis runtime.

    Host/PyTorch stages:
      - text duration
      - frame expansion
      - ProsodyPredictor.F0Ntrain
      - Decoder.decode_features
      - harmonic feature generation
      - harmonic/source pyramid generation

    TensorRT stage:
      - Generator.forward_with_source_pyramid
    """

    def __init__(
        self,
        kmodel,
        artifact_dir: Optional[Union[str, Path]] = None,
        decoder_engine: Optional[torch.nn.Module] = None,
        max_synthesis_frames: Optional[int] = None,
        fallback_to_pytorch: bool = True,
        decoder_dtype: Optional[torch.dtype] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("KokoroTRTBackend requires CUDA")

        self.device = torch.device("cuda")
        self.kmodel = kmodel.eval().to(self.device)
        self.text_duration_module = KokoroTextDuration(self.kmodel).eval()
        self.acoustic_vocoder = KokoroAcousticVocoder(self.kmodel).eval()

        self.metadata: dict[str, Any] = {}
        if artifact_dir is not None:
            artifact_dir = Path(artifact_dir)
            self.metadata = load_trt_metadata(artifact_dir)
            if decoder_engine is None:
                try:
                    import torch_tensorrt
                except ImportError as e:
                    raise ImportError(
                        "KokoroTRTBackend requires Torch-TensorRT to load compiled engines. "
                        "Install Torch-TensorRT matching your PyTorch/CUDA stack."
                    ) from e

                engine_filename = str(
                    self.metadata.get("engine_file", TRT_ENGINE_FILENAME)
                )
                engine_path = str(artifact_dir / engine_filename)
                ep = torch_tensorrt.load(engine_path)
                decoder_engine = ep.module()

        if decoder_engine is None:
            raise ValueError("decoder_engine or artifact_dir is required")

        self.generator = decoder_engine.to(self.device)
        self.fallback_to_pytorch = fallback_to_pytorch

        profile = self.metadata.get("profile", {})
        metadata_min = profile.get("min_frames")
        metadata_max = profile.get("max_frames")

        self.min_synthesis_frames = int(metadata_min if metadata_min is not None else 1)

        if max_synthesis_frames is None and metadata_max is None:
            raise ValueError(
                "max_synthesis_frames is required when artifact metadata has no profile.max_frames"
            )

        self.max_synthesis_frames = int(
            max_synthesis_frames if max_synthesis_frames is not None else metadata_max
        )

        if self.min_synthesis_frames < 1:
            raise ValueError("TensorRT profile min_frames must be positive")
        if self.max_synthesis_frames < 1:
            raise ValueError("TensorRT profile max_frames must be positive")
        if self.max_synthesis_frames < self.min_synthesis_frames:
            raise ValueError(
                "TensorRT profile max_frames must be >= min_frames, got "
                f"{self.max_synthesis_frames} < {self.min_synthesis_frames}"
            )

        precision = str(self.metadata.get("precision", "fp32")).lower()
        if decoder_dtype is not None:
            self.decoder_dtype = decoder_dtype
        elif precision == "fp16":
            self.decoder_dtype = torch.float16
        else:
            self.decoder_dtype = torch.float32

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

    def _generate_with_trt(
        self,
        x: torch.Tensor,
        ref: torch.Tensor,
        source_pyramid: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        dtype = self.decoder_dtype
        return self.generator(
            x.to(dtype=dtype),
            ref.to(dtype=dtype),
            *[source.to(dtype=dtype) for source in source_pyramid],
        ).float()

    def render(self, frame_item: FrameItem, ref_s: torch.Tensor) -> torch.Tensor:
        synthesis_frames = int(frame_item.synthesis_frame_length)
        outside_profile = (
            synthesis_frames < self.min_synthesis_frames
            or synthesis_frames > self.max_synthesis_frames
        )

        if outside_profile and not self.fallback_to_pytorch:
            raise RuntimeError(
                "Predicted synthesis frame length "
                f"{synthesis_frames} is outside the TensorRT profile "
                f"[{self.min_synthesis_frames}, {self.max_synthesis_frames}]"
            )

        asr = frame_item.asr.unsqueeze(0)
        en = frame_item.en.unsqueeze(0)

        f0, n = self.acoustic_vocoder.predict_f0n(en, ref_s)

        expected_generator_frames = self.kmodel.decoder.generator_input_frame_length(
            synthesis_frames
        )
        if int(f0.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                "F0 frame length does not match decoder/generator contract: "
                f"got {int(f0.shape[-1])}, expected {expected_generator_frames} "
                f"for synthesis frame length {synthesis_frames}"
            )
        if int(n.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                "Noise frame length does not match decoder/generator contract: "
                f"got {int(n.shape[-1])}, expected {expected_generator_frames} "
                f"for synthesis frame length {synthesis_frames}"
            )

        har = self.kmodel.compute_harmonic_features(f0)

        expected_har_frames = harmonic_frame_count(self.kmodel, synthesis_frames)
        if int(har.shape[-1]) != expected_har_frames:
            raise RuntimeError(
                "Harmonic feature frame length does not match decoder/generator "
                "contract: "
                f"got {int(har.shape[-1])}, expected {expected_har_frames} "
                f"for synthesis frame length {synthesis_frames}"
            )

        if outside_profile:
            logger.warning(
                "Predicted synthesis frame length {} is outside the TensorRT "
                "profile [{}, {}]; using PyTorch decoder fallback for this utterance.",
                synthesis_frames,
                self.min_synthesis_frames,
                self.max_synthesis_frames,
            )
            return self.acoustic_vocoder.forward_with_f0n(asr, f0, n, ref_s, har)

        source_pyramid = self.kmodel.decoder.compute_source_pyramid(
            har,
            ref_s[:, :128],
        )

        expected_source_frames = source_frame_counts(self.kmodel, synthesis_frames)
        expected_source_channels = self.kmodel.decoder.source_channels()

        if len(source_pyramid) != len(expected_source_frames):
            raise RuntimeError(
                "Source-pyramid tensor count mismatch: "
                f"got {len(source_pyramid)}, expected {len(expected_source_frames)}"
            )

        for i, source in enumerate(source_pyramid):
            if int(source.shape[1]) != int(expected_source_channels[i]):
                raise RuntimeError(
                    f"Source-pyramid tensor source_{i} channel mismatch: "
                    f"got {int(source.shape[1])}, expected "
                    f"{int(expected_source_channels[i])}"
                )
            if int(source.shape[-1]) != int(expected_source_frames[i]):
                raise RuntimeError(
                    f"Source-pyramid tensor source_{i} frame length mismatch: "
                    f"got {int(source.shape[-1])}, expected "
                    f"{int(expected_source_frames[i])} for synthesis frame length "
                    f"{synthesis_frames}"
                )

        decoded = self.kmodel.decoder.decode_features(
            asr,
            f0,
            n,
            ref_s[:, :128],
        )

        expected_generator_channels = generator_input_channels(self.kmodel)
        if int(decoded.shape[1]) != expected_generator_channels:
            raise RuntimeError(
                "Decoded generator input channel mismatch: "
                f"got {int(decoded.shape[1])}, expected "
                f"{expected_generator_channels}"
            )

        if int(decoded.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                "Decoded generator input frame length mismatch: "
                f"got {int(decoded.shape[-1])}, expected "
                f"{expected_generator_frames} for synthesis frame length "
                f"{synthesis_frames}"
            )

        return self._generate_with_trt(decoded, ref_s, source_pyramid)

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
