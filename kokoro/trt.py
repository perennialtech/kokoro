import threading
from pathlib import Path
from typing import Generator, Literal, Optional, Union

import torch

from .artifact import TensorRTArtifact
from .graphs import RenderGraphs, TextDurationGraphs
from .model import KokoroHostStages, KokoroModelLoader
from .native_trt import NativeTRTEngine
from .pipeline import TextFrontend, VoiceStore
from .runtime import synthesize_prepared_trt
from .shapes import ShapePlan
from .telemetry import (
    NoOpProfileContext,
    ProfileContext,
    Telemetry,
    shape_attr,
    tensor_nbytes,
)
from .types import FrameItem, SynthesisResult


class KokoroTRT:
    def __init__(
        self,
        artifact_dir: Union[str, Path],
        voice_dir: Optional[Union[str, Path]] = None,
        verify_internal_shapes: bool = False,
        telemetry: Optional[Telemetry] = None,
        text_graph_max_tokens: int = 96,
        render_graph_max_frames: int = 160,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("KokoroTRT requires CUDA")
        if text_graph_max_tokens < 0 or render_graph_max_frames < 0:
            raise ValueError("graph limits must be non-negative")

        self.telemetry = telemetry or Telemetry()
        self.artifact = TensorRTArtifact.load(artifact_dir)
        self.artifact.validate_gpu()
        self.metadata = self.artifact.metadata
        self.telemetry.register_runtime(self.metadata)

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

        self.host = KokoroHostStages(model).to(self.device).eval()
        self.generator = NativeTRTEngine(self.artifact.paths.engine_path)
        if set(self.generator.input_names) != set(self.metadata.input_names):
            raise ValueError("TensorRT engine inputs do not match artifact metadata")
        if self.generator.output_names != self.metadata.output_names:
            raise ValueError("TensorRT engine outputs do not match artifact metadata")
        for name in self.generator.input_names:
            if self.generator.tensor_dtypes[name] != torch.float16:
                raise ValueError(f"TensorRT generator input {name!r} is not FP16")
        if self.generator.tensor_dtypes["audio"] != torch.float32:
            raise ValueError("TensorRT generator output must be FP32")
        self._inference_lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self.device)
        self._text_graph_max_tokens = min(
            text_graph_max_tokens, self.host.context_length
        )
        self._render_graph_max_frames = min(
            render_graph_max_frames, self.host.model.predictor.shared.max_seq_len
        )
        self._text_graphs: Optional[TextDurationGraphs] = None
        self._render_graphs: Optional[RenderGraphs] = None
        self._graph_math_mode: Optional[tuple] = None

        self.shape_plan = ShapePlan.from_model(self.host.model)

        self.voice_store = VoiceStore(
            Path(voice_dir) if voice_dir is not None else self.artifact.paths.voice_dir
        )
        self.voice_store.set_target(self.device, torch.float32)
        self._frontends: dict[Literal["zh", "ja"], TextFrontend] = {}

    def _ensure_graphs(self):
        mode = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
        )
        if mode == self._graph_math_mode:
            return
        # Captured kernels freeze math choices. Production may enable TF32 after
        # constructing Kokoro, so build only at warmup and rebuild on changes.
        self._inference_stream.synchronize()
        self._graph_math_mode = None
        self._text_graphs = None
        self._render_graphs = None
        if self._text_graph_max_tokens >= 3:
            self._text_graphs = TextDurationGraphs(
                self.host, self._text_graph_max_tokens
            )
        if self._render_graph_max_frames >= 2:
            self._render_graphs = RenderGraphs(self.host, self._render_graph_max_frames)
        self._graph_math_mode = mode

    def warmup(self) -> None:
        """Capture bounded host graphs using the active PyTorch math settings."""
        with self._inference_lock:
            caller = torch.cuda.current_stream(self.device)
            self._inference_stream.wait_stream(caller)
            with torch.cuda.stream(self._inference_stream):
                self._ensure_graphs()
            caller.wait_stream(self._inference_stream)

    def frontend(
        self,
        default_han_language: Literal["zh", "ja"],
    ) -> TextFrontend:
        if default_han_language not in ("zh", "ja"):
            raise ValueError("default_han_language must be either 'zh' or 'ja'")

        cached = self._frontends.get(default_han_language)
        if cached is not None:
            return cached

        if self.host.vocab is None:
            raise ValueError("Artifact config does not contain a vocab")

        frontend = TextFrontend(
            default_han_language=default_han_language,
            vocab=self.host.vocab,
            context_length=self.host.context_length,
            voice_store=self.voice_store,
        )
        self._frontends[default_han_language] = frontend
        return frontend

    def prepare(
        self,
        text: Union[str, list[str]],
        voice: str,
        default_han_language: Literal["zh", "ja"],
        speed: float = 1.0,
        split_pattern: Optional[str] = r"\n+",
        profile: Optional[ProfileContext] = None,
    ):
        yield from self.frontend(default_han_language).prepare(
            text=text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
            profile=profile,
        )

    def synthesize_prepared(
        self,
        prepared,
        profile: Optional[ProfileContext] = None,
    ) -> SynthesisResult:
        # Host LSTMs and text graphs have mutable workspaces. Enqueue complete
        # chunks on one stream so callers on different threads cannot race.
        with self._inference_lock:
            caller = torch.cuda.current_stream(self.device)
            self._inference_stream.wait_stream(caller)
            with torch.cuda.stream(self._inference_stream):
                self._ensure_graphs()
                for tensor in (prepared.input_ids, prepared.ref_s, prepared.speed):
                    if tensor.is_cuda:
                        tensor.record_stream(self._inference_stream)
                result = synthesize_prepared_trt(self, prepared, profile=profile)
            caller.wait_stream(self._inference_stream)
            for tensor in (result.audio, result.pred_dur, result.duration_float):
                tensor.record_stream(caller)
            return result

    def _text_duration(self, input_ids, ref_s, speed):
        if self._text_graphs is not None:
            return self._text_graphs(input_ids, ref_s, speed)
        return self.host.text_duration(input_ids, ref_s, speed)

    def synthesize(
        self,
        text: Union[str, list[str]],
        voice: str,
        default_han_language: Literal["zh", "ja"],
        speed: float = 1.0,
        split_pattern: Optional[str] = r"\n+",
    ) -> Generator[SynthesisResult, None, None]:
        request = self.telemetry.start_request(
            language=default_han_language,
            voice=voice,
            speed=speed,
            input_chars=(
                sum(len(x) for x in text) if isinstance(text, list) else len(text)
            ),
            precision=self.metadata.precision,
        )
        status = "cancelled"
        error: BaseException | None = None

        try:
            for prepared in self.prepare(
                text=text,
                voice=voice,
                default_han_language=default_han_language,
                speed=speed,
                split_pattern=split_pattern,
                profile=request,
            ):
                yield self.synthesize_prepared(prepared, profile=request)
            status = "ok"
        except GeneratorExit:
            status = "cancelled"
            raise
        except Exception as e:
            status = "error"
            error = e
            raise
        finally:
            request.finalize(status, error)

    def _generate_with_trt(
        self,
        x: torch.Tensor,
        ref_s: torch.Tensor,
        source_pyramid: tuple[torch.Tensor, ...],
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        dtype = torch.float16

        with profile.span(
            "trt.prepare_inputs",
            attrs={
                "precision": self.metadata.precision,
                "source_count": len(source_pyramid),
            },
        ):
            raw_inputs = {"x": x, "ref_s": ref_s}
            raw_inputs.update(
                {f"source_{i}": source for i, source in enumerate(source_pyramid)}
            )

        with profile.span("trt.input_cast", cuda=True) as span:
            inputs = {
                name: tensor.to(dtype=dtype).contiguous()
                for name, tensor in raw_inputs.items()
            }
            total_input_bytes = 0
            for name, tensor in inputs.items():
                nbytes = tensor_nbytes(tensor)
                total_input_bytes += nbytes
                span.attrs[f"{name}.shape"] = shape_attr(tensor)
                span.attrs[f"{name}.dtype"] = str(tensor.dtype)
                span.attrs[f"{name}.bytes"] = nbytes
            span.attrs["input_bytes"] = total_input_bytes
            profile.histogram("trt_input_bytes", float(total_input_bytes), {})

        with profile.span("trt.run", cuda=True):
            outputs = self.generator.run(inputs, profile=profile)

        with profile.span("trt.output_cast", cuda=True) as span:
            audio = outputs["audio"].float()
            span.attrs["audio.shape"] = shape_attr(audio)
            span.attrs["audio.bytes"] = tensor_nbytes(audio)
            return audio

    def render_frame(
        self,
        frame_item: FrameItem,
        ref_s: torch.Tensor,
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        synthesis_frames = int(frame_item.synthesis_frame_length)

        asr = frame_item.asr.unsqueeze(0).to(self.device)
        en = frame_item.en.unsqueeze(0).to(self.device)
        ref_s = ref_s.to(self.device, dtype=torch.float32)

        if (
            self._render_graphs is not None
            and synthesis_frames in self._render_graphs.graphs
        ):
            with profile.span("host.render_graph", cuda=True) as span:
                f0, n, har, decoded, *sources = self._render_graphs(asr, en, ref_s)
                source_pyramid = tuple(sources)
                span.attrs["synthesis_frames"] = synthesis_frames
        else:
            with profile.span("host.predict_f0n", cuda=True) as span:
                f0, n = self.host.predict_f0n(en, ref_s)
                span.attrs["f0.shape"] = shape_attr(f0)
                span.attrs["noise.shape"] = shape_attr(n)

            with profile.span("host.compute_harmonic_features", cuda=True) as span:
                har = self.host.compute_harmonic_features(f0)
                span.attrs["har.shape"] = shape_attr(har)

            with profile.span("host.compute_source_pyramid", cuda=True) as span:
                source_pyramid = self.host.compute_source_pyramid(har, ref_s)
                for i, source in enumerate(source_pyramid):
                    span.attrs[f"source_{i}.shape"] = shape_attr(source)

            with profile.span("host.decode_features", cuda=True) as span:
                decoded = self.host.decode_features(asr, f0, n, ref_s)
                span.attrs["decoded.shape"] = shape_attr(decoded)

        if self.verify_internal_shapes:
            with profile.span("runtime.verify_internal_shapes"):
                self._verify_internal_contract(
                    synthesis_frames=synthesis_frames,
                    f0=f0,
                    noise=n,
                    har=har,
                    source_pyramid=source_pyramid,
                    decoded=decoded,
                )

        with profile.span("trt.generator", cuda=True):
            return self._generate_with_trt(
                decoded, ref_s, source_pyramid, profile=profile
            )

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
