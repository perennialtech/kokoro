from pathlib import Path
from typing import Generator, Optional, Union

import torch

from .artifact import TensorRTArtifact
from .model import KokoroHostStages, KokoroModelLoader
from .native_trt import NativeTRTEngine
from .pipeline import TextFrontend, VoiceStore, normalize_language_code
from .runtime import synthesize_prepared_trt
from .shapes import ShapePlan
from .types import FrameItem, SynthesisResult


class KokoroTRT:
    def __init__(
        self,
        artifact_dir: Union[str, Path],
        voice_dir: Optional[Union[str, Path]] = None,
        verify_internal_shapes: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("KokoroTRT requires CUDA")

        self.artifact = TensorRTArtifact.load(artifact_dir)
        self.artifact.validate_gpu()
        self.metadata = self.artifact.metadata

        self.device = torch.device("cuda")
        self.verify_internal_shapes = bool(verify_internal_shapes)

        config_data = self.artifact.load_config()
        loader = KokoroModelLoader(
            repo_id=self.metadata.repo_id,
            config=config_data,
            model=None,
        )
        model = loader.load(load_weights=False)
        model.load_host_state(self.artifact.paths.host_state_path)

        self.host = KokoroHostStages(model).to(self.device)

        self.generator = NativeTRTEngine(self.artifact.paths.engine_path)

        self.decoder_dtype = (
            torch.float16 if self.metadata.precision == "fp16" else torch.float32
        )
        self.profile = self.metadata.profile
        self.min_synthesis_frames = self.profile.min_frames
        self.max_synthesis_frames = self.profile.max_frames
        self.shape_plan = ShapePlan.from_model(self.host.model, self.profile)

        self.voice_store = VoiceStore(
            Path(voice_dir) if voice_dir is not None else self.artifact.paths.voice_dir
        )
        self.voice_store.set_target(self.device, torch.float32)
        self._frontends: dict[str, TextFrontend] = {}

    def frontend(self, lang_code: str) -> TextFrontend:
        lang_code = normalize_language_code(lang_code)
        cached = self._frontends.get(lang_code)
        if cached is not None:
            return cached

        if self.host.vocab is None:
            raise ValueError("Artifact config does not contain a vocab")

        frontend = TextFrontend(
            lang_code=lang_code,
            repo_id=self.host.repo_id,
            vocab=self.host.vocab,
            context_length=self.host.context_length,
            voice_store=self.voice_store,
        )
        self._frontends[lang_code] = frontend
        return frontend

    def prepare(
        self,
        text: Union[str, list[str]],
        voice: Union[str, torch.Tensor],
        language: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = r"\n+",
    ):
        yield from self.frontend(language).prepare(
            text=text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
        )

    def synthesize_prepared(self, prepared) -> SynthesisResult:
        return synthesize_prepared_trt(self, prepared)

    def synthesize(
        self,
        text: Union[str, list[str]],
        voice: Union[str, torch.Tensor],
        language: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator[SynthesisResult, None, None]:
        for prepared in self.prepare(
            text=text,
            voice=voice,
            language=language,
            speed=speed,
            split_pattern=split_pattern,
        ):
            yield self.synthesize_prepared(prepared)

    def _generate_with_trt(
        self,
        x: torch.Tensor,
        ref_s: torch.Tensor,
        source_pyramid: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        dtype = self.decoder_dtype
        inputs = {
            "x": x.to(dtype=dtype).contiguous(),
            "ref_s": ref_s.to(dtype=dtype).contiguous(),
        }
        inputs.update(
            {
                f"source_{i}": source.to(dtype=dtype).contiguous()
                for i, source in enumerate(source_pyramid)
            }
        )
        return self.generator.run(inputs)["audio"].float()

    def render_frame(self, frame_item: FrameItem, ref_s: torch.Tensor) -> torch.Tensor:
        synthesis_frames = int(frame_item.synthesis_frame_length)
        if (
            synthesis_frames < self.min_synthesis_frames
            or synthesis_frames > self.max_synthesis_frames
        ):
            raise RuntimeError(
                "Predicted synthesis frame length "
                f"{synthesis_frames} is outside the TensorRT profile "
                f"[{self.min_synthesis_frames}, {self.max_synthesis_frames}]. "
                "Recompile with a wider --min-frames/--max-frames profile."
            )

        asr = frame_item.asr.unsqueeze(0).to(self.device)
        en = frame_item.en.unsqueeze(0).to(self.device)
        ref_s = ref_s.to(self.device, dtype=torch.float32)

        f0, n = self.host.predict_f0n(en, ref_s)
        har = self.host.compute_harmonic_features(f0)
        source_pyramid = self.host.compute_source_pyramid(har, ref_s)
        decoded = self.host.decode_features(asr, f0, n, ref_s)

        if self.verify_internal_shapes:
            self._verify_internal_contract(
                synthesis_frames=synthesis_frames,
                f0=f0,
                noise=n,
                har=har,
                source_pyramid=source_pyramid,
                decoded=decoded,
            )

        return self._generate_with_trt(decoded, ref_s, source_pyramid)

    def _verify_internal_contract(
        self,
        *,
        synthesis_frames: int,
        f0: torch.Tensor,
        noise: torch.Tensor,
        har: torch.Tensor,
        source_pyramid: tuple[torch.Tensor, ...],
        decoded: torch.Tensor,
    ) -> None:
        expected_generator_frames = self.shape_plan.generator_frames(
            self.host.model,
            synthesis_frames,
        )
        if int(f0.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                f"F0 frame length mismatch: got {int(f0.shape[-1])}, "
                f"expected {expected_generator_frames}"
            )
        if int(noise.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                f"Noise frame length mismatch: got {int(noise.shape[-1])}, "
                f"expected {expected_generator_frames}"
            )
        if int(decoded.shape[-1]) != expected_generator_frames:
            raise RuntimeError(
                f"Decoded frame length mismatch: got {int(decoded.shape[-1])}, "
                f"expected {expected_generator_frames}"
            )
        if int(decoded.shape[1]) != self.shape_plan.input_channels:
            raise RuntimeError(
                f"Decoded channel mismatch: got {int(decoded.shape[1])}, "
                f"expected {self.shape_plan.input_channels}"
            )

        expected_har_frames = self.shape_plan.harmonic_frames(
            self.host.model,
            synthesis_frames,
        )
        if int(har.shape[-1]) != expected_har_frames:
            raise RuntimeError(
                f"Harmonic frame length mismatch: got {int(har.shape[-1])}, "
                f"expected {expected_har_frames}"
            )

        expected_source_lengths = self.shape_plan.source_lengths(
            self.host.model,
            synthesis_frames,
        )
        if len(source_pyramid) != len(expected_source_lengths):
            raise RuntimeError(
                f"Source-pyramid tensor count mismatch: got {len(source_pyramid)}, "
                f"expected {len(expected_source_lengths)}"
            )

        for i, source in enumerate(source_pyramid):
            expected_channels = self.shape_plan.source_channels[i]
            expected_frames = expected_source_lengths[i]
            if int(source.shape[1]) != expected_channels:
                raise RuntimeError(
                    f"source_{i} channel mismatch: got {int(source.shape[1])}, "
                    f"expected {expected_channels}"
                )
            if int(source.shape[-1]) != expected_frames:
                raise RuntimeError(
                    f"source_{i} frame mismatch: got {int(source.shape[-1])}, "
                    f"expected {expected_frames}"
                )
