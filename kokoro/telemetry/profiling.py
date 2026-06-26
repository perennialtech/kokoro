from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

from .metrics import MetricsSink, NoOpMetrics

AttrValue = Union[int, float, str, bool, None]
Status = Literal["ok", "error", "cancelled"]


@dataclass
class ProfilerConfig:
    enabled: bool = False
    cuda_timing: bool = False
    synchronize_cuda: bool = False
    emit_nvtx: bool = False
    include_text: bool = False
    sample_rate: float = 1.0
    record_shapes: bool = True
    record_memory: bool = True
    record_stage_metrics: bool = True


@dataclass
class StageRecord:
    name: str
    parent: Optional[str]
    cpu_start_ns: int
    cpu_end_ns: int
    cpu_ms: float
    cuda_ms: Optional[float]
    attrs: dict[str, AttrValue] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ChunkTrace:
    schema_version: int = 1
    kind: str = "chunk"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = ""
    chunk_index: int = 0
    status: Status = "ok"
    language: str = ""
    voice_label: str = ""
    voice_kind: str = ""
    speed: float = 1.0
    precision: str = ""
    profile_min_frames: int = 0
    profile_opt_frames: int = 0
    profile_max_frames: int = 0
    input_chars: int = 0
    phoneme_chars: int = 0
    phoneme_ids: int = 0
    input_ids: int = 0
    synthesis_frames: int = 0
    return_frames: int = 0
    sample_length: int = 0
    audio_duration_s: float = 0.0
    submit_latency_s: float = 0.0
    ready_latency_s: Optional[float] = None
    rtf_submit: Optional[float] = None
    rtf_ready: Optional[float] = None
    stages: list[StageRecord] = field(default_factory=list)
    graphemes: Optional[str] = None
    phonemes: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RequestTrace:
    schema_version: int = 1
    kind: str = "request"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: Status = "ok"
    language: str = ""
    voice_label: str = ""
    voice_kind: str = ""
    speed: float = 1.0
    precision: str = ""
    input_chars: int = 0
    chunks_total: int = 0
    chunks_ok: int = 0
    chunks_error: int = 0
    audio_duration_s: float = 0.0
    submit_latency_s: float = 0.0
    ready_latency_s: Optional[float] = None
    rtf_submit: Optional[float] = None
    rtf_ready: Optional[float] = None
    stages: list[StageRecord] = field(default_factory=list)
    chunks: list[ChunkTrace] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class SpanHandle:
    def __init__(self, attrs: Optional[dict[str, AttrValue]] = None):
        self.attrs = attrs if attrs is not None else {}


class _NoOpSpan(SpanHandle):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class ProfileContext:
    enabled = False

    def span(
        self,
        name: str,
        *,
        cuda: bool = False,
        attrs: Optional[dict[str, AttrValue]] = None,
    ):
        return _NoOpSpan(attrs)

    def counter(self, name: str, value: float = 1, labels: dict[str, str] = {}) -> None:
        return

    def histogram(self, name: str, value: float, labels: dict[str, str] = {}) -> None:
        return

    def gauge(self, name: str, value: float, labels: dict[str, str] = {}) -> None:
        return

    def sample_gpu_memory(self, suffix: str = "") -> None:
        return


class NoOpProfileContext(ProfileContext):
    pass


class NoOpRequestProfile(NoOpProfileContext):
    trace = RequestTrace(status="cancelled")

    def start_chunk(self, prepared: Any = None):
        return NoOpChunkProfile()

    def finalize(
        self, status: Status = "ok", error: BaseException | None = None
    ) -> None:
        return


class NoOpChunkProfile(NoOpProfileContext):
    trace = ChunkTrace(status="cancelled")

    def finalize(
        self, status: Status = "ok", error: BaseException | None = None
    ) -> None:
        return


class _Span:
    def __init__(
        self,
        profile: "_RecordingProfileContext",
        name: str,
        cuda: bool,
        attrs: Optional[dict[str, AttrValue]],
    ):
        self.profile = profile
        self.name = name
        self.cuda = bool(cuda)
        self.attrs = attrs if attrs is not None else {}
        self.parent: Optional[str] = None
        self.cpu_start_ns = 0
        self.start_event = None
        self.end_event = None
        self.nvtx = False

    def __enter__(self):
        self.parent = self.profile._stack[-1] if self.profile._stack else None
        self.profile._stack.append(self.name)
        self.cpu_start_ns = time.perf_counter_ns()

        if self.profile.config.emit_nvtx:
            try:
                import torch

                torch.cuda.nvtx.range_push(self.name)
                self.nvtx = True
            except Exception:
                self.nvtx = False

        if self.cuda and self.profile.config.cuda_timing:
            try:
                import torch

                if torch.cuda.is_available():
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
                    self.start_event.record()
            except Exception:
                self.start_event = None
                self.end_event = None

        return self

    def __exit__(self, exc_type, exc, tb):
        if self.end_event is not None:
            self.end_event.record()

        cpu_end_ns = time.perf_counter_ns()

        if self.nvtx:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except Exception:
                pass

        if self.profile._stack and self.profile._stack[-1] == self.name:
            self.profile._stack.pop()

        record = StageRecord(
            name=self.name,
            parent=self.parent,
            cpu_start_ns=self.cpu_start_ns,
            cpu_end_ns=cpu_end_ns,
            cpu_ms=(cpu_end_ns - self.cpu_start_ns) / 1_000_000.0,
            cuda_ms=None,
            attrs={k: _attr(v) for k, v in self.attrs.items()},
            error_type=None if exc_type is None else exc_type.__name__,
            error_message=None if exc is None else str(exc),
        )
        self.profile._record_stage(record, self.start_event, self.end_event)
        return False


class _RecordingProfileContext(ProfileContext):
    enabled = True

    def __init__(self, telemetry: "Telemetry"):
        self.telemetry = telemetry
        self.config = telemetry.profiler_config
        self._stack: list[str] = []
        self._pending_cuda: list[tuple[StageRecord, Any, Any]] = []

    def span(
        self,
        name: str,
        *,
        cuda: bool = False,
        attrs: Optional[dict[str, AttrValue]] = None,
    ):
        if not self.telemetry.active:
            return _NoOpSpan(attrs)
        return _Span(self, name, cuda, attrs)

    def _base_labels(self) -> dict[str, str]:
        return {}

    def _stages(self) -> list[StageRecord]:
        raise NotImplementedError

    def _record_stage(
        self, record: StageRecord, start_event: Any, end_event: Any
    ) -> None:
        self._stages().append(record)

        labels = {
            **self._base_labels(),
            "stage": record.name,
            "status": "error" if record.error_type else "ok",
        }
        if self.config.record_stage_metrics:
            self.telemetry.metrics.observe_histogram(
                "stage_latency_seconds", record.cpu_ms / 1000.0, labels
            )
        if record.error_type:
            self.telemetry.metrics.observe_counter(
                "errors_total",
                1,
                {
                    "stage": record.name,
                    "error_type": record.error_type,
                    "language": labels.get("language", ""),
                    "precision": labels.get("precision", ""),
                },
            )
        if start_event is not None and end_event is not None:
            self._pending_cuda.append((record, start_event, end_event))

    def _finalize_cuda(self) -> None:
        if not self._pending_cuda or not self.config.synchronize_cuda:
            return

        try:
            import torch

            torch.cuda.current_stream().synchronize()
            for record, start_event, end_event in self._pending_cuda:
                record.cuda_ms = float(start_event.elapsed_time(end_event))
                if self.config.record_stage_metrics:
                    self.telemetry.metrics.observe_histogram(
                        "stage_cuda_seconds",
                        record.cuda_ms / 1000.0,
                        {
                            **self._base_labels(),
                            "stage": record.name,
                            "status": "error" if record.error_type else "ok",
                        },
                    )
        finally:
            self._pending_cuda.clear()

    def counter(self, name: str, value: float = 1, labels: dict[str, str] = {}) -> None:
        self.telemetry.metrics.observe_counter(name, value, labels)

    def histogram(self, name: str, value: float, labels: dict[str, str] = {}) -> None:
        self.telemetry.metrics.observe_histogram(name, value, labels)

    def gauge(self, name: str, value: float, labels: dict[str, str] = {}) -> None:
        self.telemetry.metrics.set_gauge(name, value, labels)

    def sample_gpu_memory(self, suffix: str = "") -> None:
        self.telemetry.sample_gpu_memory(suffix)


class RequestProfile(_RecordingProfileContext):
    def __init__(
        self,
        telemetry: "Telemetry",
        *,
        language: str = "",
        voice: Any = None,
        speed: float = 1.0,
        input_chars: int = 0,
        precision: str = "",
    ):
        super().__init__(telemetry)
        label, kind = normalize_voice_label(voice)
        self.trace = RequestTrace(
            status="cancelled",
            language=language,
            voice_label=label,
            voice_kind=kind,
            speed=float(speed),
            input_chars=int(input_chars),
            precision=precision or telemetry.runtime_metadata.get("precision", ""),
        )
        self._start_ns = time.perf_counter_ns()
        self._finalized = False
        self.sample_gpu_memory("request_start")

    def _stages(self) -> list[StageRecord]:
        return self.trace.stages

    def _base_labels(self) -> dict[str, str]:
        return {
            "language": self.trace.language,
            "precision": self.trace.precision,
            "voice_kind": self.trace.voice_kind,
        }

    def start_chunk(self, prepared: Any = None) -> "ChunkProfile":
        return ChunkProfile(
            self.telemetry,
            request=self,
            prepared=prepared,
            chunk_index=self.trace.chunks_total,
        )

    def _record_chunk(self, chunk: ChunkTrace) -> None:
        self.trace.chunks_total += 1
        if chunk.status == "ok":
            self.trace.chunks_ok += 1
        else:
            self.trace.chunks_error += 1
        self.trace.audio_duration_s += float(chunk.audio_duration_s)
        self.trace.chunks.append(chunk)

    def finalize(
        self, status: Status = "ok", error: BaseException | None = None
    ) -> None:
        if self._finalized:
            return
        self._finalized = True

        submit_end_ns = time.perf_counter_ns()
        self.trace.submit_latency_s = (submit_end_ns - self._start_ns) / 1_000_000_000.0
        self.trace.status = status
        if error is not None:
            self.trace.error_type = type(error).__name__
            self.trace.error_message = str(error)

        self._finalize_cuda()
        ready_end_ns = time.perf_counter_ns()
        if self.config.synchronize_cuda:
            self.trace.ready_latency_s = (
                ready_end_ns - self._start_ns
            ) / 1_000_000_000.0

        if self.trace.audio_duration_s > 0:
            self.trace.rtf_submit = (
                self.trace.submit_latency_s / self.trace.audio_duration_s
            )
            if self.trace.ready_latency_s is not None:
                self.trace.rtf_ready = (
                    self.trace.ready_latency_s / self.trace.audio_duration_s
                )

        self.sample_gpu_memory("request_finish")
        labels = {**self._base_labels(), "status": self.trace.status}
        self.telemetry.metrics.observe_counter("requests_total", 1, labels)
        self.telemetry.metrics.observe_histogram(
            "request_submit_latency_seconds", self.trace.submit_latency_s, labels
        )
        if self.trace.ready_latency_s is not None:
            self.telemetry.metrics.observe_histogram(
                "request_ready_latency_seconds", self.trace.ready_latency_s, labels
            )
        if self.trace.rtf_submit is not None:
            self.telemetry.metrics.observe_histogram(
                "rtf_submit", self.trace.rtf_submit, labels
            )
        if self.trace.rtf_ready is not None:
            self.telemetry.metrics.observe_histogram(
                "rtf_ready", self.trace.rtf_ready, labels
            )

        self.telemetry.emit_trace(self.trace)


class ChunkProfile(_RecordingProfileContext):
    def __init__(
        self,
        telemetry: "Telemetry",
        *,
        request: RequestProfile | None = None,
        prepared: Any = None,
        chunk_index: int = 0,
    ):
        super().__init__(telemetry)
        self.request = request
        self.trace = ChunkTrace(
            status="cancelled",
            request_id="" if request is None else request.trace.request_id,
            chunk_index=int(chunk_index),
            language="" if request is None else request.trace.language,
            voice_label="" if request is None else request.trace.voice_label,
            voice_kind="" if request is None else request.trace.voice_kind,
            speed=1.0 if request is None else request.trace.speed,
            precision=telemetry.runtime_metadata.get("precision", ""),
            profile_min_frames=int(
                telemetry.runtime_metadata.get("profile_min_frames", 0) or 0
            ),
            profile_opt_frames=int(
                telemetry.runtime_metadata.get("profile_opt_frames", 0) or 0
            ),
            profile_max_frames=int(
                telemetry.runtime_metadata.get("profile_max_frames", 0) or 0
            ),
        )
        if prepared is not None:
            self.update_from_prepared(prepared)
        self._start_ns = time.perf_counter_ns()
        self._finalized = False

    def update_from_prepared(self, prepared: Any) -> None:
        graphemes = getattr(prepared, "graphemes", None) or ""
        phonemes = getattr(prepared, "phonemes", None) or ""
        self.trace.input_chars = len(graphemes)
        self.trace.phoneme_chars = len(phonemes)
        self.trace.phoneme_ids = max(0, int(getattr(prepared, "input_length", 0)) - 2)
        self.trace.input_ids = int(getattr(prepared, "input_length", 0))
        if self.config.include_text:
            self.trace.graphemes = graphemes
            self.trace.phonemes = phonemes

    def _stages(self) -> list[StageRecord]:
        return self.trace.stages

    def _base_labels(self) -> dict[str, str]:
        return {
            "language": self.trace.language,
            "precision": self.trace.precision,
            "voice_kind": self.trace.voice_kind,
        }

    def finalize(
        self, status: Status = "ok", error: BaseException | None = None
    ) -> None:
        if self._finalized:
            return
        self._finalized = True

        submit_end_ns = time.perf_counter_ns()
        self.trace.submit_latency_s = (submit_end_ns - self._start_ns) / 1_000_000_000.0
        self.trace.status = status
        if error is not None:
            self.trace.error_type = type(error).__name__
            self.trace.error_message = str(error)

        self._finalize_cuda()
        ready_end_ns = time.perf_counter_ns()
        if self.config.synchronize_cuda:
            self.trace.ready_latency_s = (
                ready_end_ns - self._start_ns
            ) / 1_000_000_000.0

        if self.trace.audio_duration_s > 0:
            self.trace.rtf_submit = (
                self.trace.submit_latency_s / self.trace.audio_duration_s
            )
            if self.trace.ready_latency_s is not None:
                self.trace.rtf_ready = (
                    self.trace.ready_latency_s / self.trace.audio_duration_s
                )

        self.sample_gpu_memory("chunk_finish")
        labels = {**self._base_labels(), "status": self.trace.status}
        self.telemetry.metrics.observe_counter("chunks_total", 1, labels)
        self.telemetry.metrics.observe_histogram(
            "chunk_submit_latency_seconds", self.trace.submit_latency_s, labels
        )
        if self.trace.ready_latency_s is not None:
            self.telemetry.metrics.observe_histogram(
                "chunk_ready_latency_seconds", self.trace.ready_latency_s, labels
            )
        self.telemetry.metrics.observe_histogram(
            "audio_duration_seconds", self.trace.audio_duration_s, labels
        )
        for name, value in (
            ("input_chars", self.trace.input_chars),
            ("phoneme_chars", self.trace.phoneme_chars),
            ("phoneme_ids", self.trace.phoneme_ids),
            ("input_ids", self.trace.input_ids),
            ("synthesis_frames", self.trace.synthesis_frames),
            ("return_frames", self.trace.return_frames),
            ("sample_length", self.trace.sample_length),
        ):
            self.telemetry.metrics.observe_histogram(name, float(value), labels)
        if self.trace.rtf_submit is not None:
            self.telemetry.metrics.observe_histogram(
                "rtf_submit", self.trace.rtf_submit, labels
            )
        if self.trace.rtf_ready is not None:
            self.telemetry.metrics.observe_histogram(
                "rtf_ready", self.trace.rtf_ready, labels
            )

        self.telemetry.emit_trace(self.trace)
        if self.request is not None:
            self.request._record_chunk(self.trace)


class Telemetry:
    def __init__(
        self,
        profiler_config: ProfilerConfig | None = None,
        trace_sinks: Optional[list[Any]] = None,
        metrics: MetricsSink | None = None,
    ):
        self.profiler_config = profiler_config or ProfilerConfig()
        self.trace_sinks = trace_sinks or []
        self.metrics: MetricsSink = metrics or NoOpMetrics()
        self.runtime_metadata: dict[str, Any] = {}
        self.last_request_trace: RequestTrace | None = None
        self.last_chunk_trace: ChunkTrace | None = None
        self.active = (
            self.profiler_config.enabled
            or bool(self.trace_sinks)
            or not isinstance(self.metrics, NoOpMetrics)
        )

    def start_request(
        self,
        *,
        language: str = "",
        voice: Any = None,
        speed: float = 1.0,
        input_chars: int = 0,
        precision: str = "",
    ) -> RequestProfile | NoOpRequestProfile:
        if not self.active:
            return NoOpRequestProfile()
        return RequestProfile(
            self,
            language=language,
            voice=voice,
            speed=speed,
            input_chars=input_chars,
            precision=precision,
        )

    def start_chunk(
        self,
        *,
        prepared: Any = None,
        request: RequestProfile | None = None,
        chunk_index: int = 0,
    ) -> ChunkProfile | NoOpChunkProfile:
        if not self.active:
            return NoOpChunkProfile()
        return ChunkProfile(
            self, request=request, prepared=prepared, chunk_index=chunk_index
        )

    def emit_trace(self, trace: ChunkTrace | RequestTrace) -> None:
        if isinstance(trace, RequestTrace):
            self.last_request_trace = trace
        elif isinstance(trace, ChunkTrace):
            self.last_chunk_trace = trace

        for sink in self.trace_sinks:
            sink.emit_trace(trace)

    def register_runtime(self, metadata: Any) -> None:
        gpu = getattr(metadata, "gpu", {}) or {}
        profile = getattr(metadata, "profile", None)
        versions = getattr(metadata, "versions", {}) or {}

        self.runtime_metadata = {
            "artifact_type": getattr(metadata, "artifact_type", ""),
            "format_version": getattr(metadata, "format_version", ""),
            "repo_id": getattr(metadata, "repo_id", ""),
            "precision": getattr(metadata, "precision", ""),
            "tensorrt_version": versions.get("tensorrt"),
            "torch_version": versions.get("torch"),
            "gpu_name": gpu.get("name", ""),
            "compute_capability": gpu.get("compute_capability", ""),
            "profile_min_frames": getattr(profile, "min_frames", 0),
            "profile_opt_frames": getattr(profile, "opt_frames", 0),
            "profile_max_frames": getattr(profile, "max_frames", 0),
            "input_names": ",".join(getattr(metadata, "input_names", ()) or ()),
            "output_names": ",".join(getattr(metadata, "output_names", ()) or ()),
        }

        self.metrics.set_info(
            "artifact_info",
            {
                "repo_id": str(self.runtime_metadata["repo_id"]),
                "precision": str(self.runtime_metadata["precision"]),
                "compute_capability": str(self.runtime_metadata["compute_capability"]),
                "format_version": str(self.runtime_metadata["format_version"]),
            },
        )
        for bound in ("min", "opt", "max"):
            self.metrics.set_gauge(
                "profile_frames",
                float(self.runtime_metadata.get(f"profile_{bound}_frames", 0) or 0),
                {"bound": bound},
            )

    def sample_gpu_memory(self, suffix: str = "") -> None:
        if not self.profiler_config.record_memory:
            return
        try:
            import torch

            if not torch.cuda.is_available():
                return
            labels = {"sample": suffix} if suffix else {}
            self.metrics.set_gauge(
                "cuda_memory_allocated_bytes",
                float(torch.cuda.memory_allocated()),
                labels,
            )
            self.metrics.set_gauge(
                "cuda_memory_reserved_bytes",
                float(torch.cuda.memory_reserved()),
                labels,
            )
            self.metrics.set_gauge(
                "cuda_max_memory_allocated_bytes",
                float(torch.cuda.max_memory_allocated()),
                labels,
            )
        except Exception:
            return


def _attr(value: Any) -> AttrValue:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def shape_attr(tensor: Any) -> str:
    return "x".join(str(int(dim)) for dim in getattr(tensor, "shape", ()))


def tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def normalize_voice_label(voice: Any) -> tuple[str, str]:
    if voice is None:
        return "", ""
    try:
        import torch

        if isinstance(voice, torch.Tensor):
            return "tensor", "tensor"
    except Exception:
        pass

    if not isinstance(voice, str):
        return "external", "external"

    parts = [p.strip() for p in voice.split(",") if p.strip()]
    if len(parts) > 1:
        return "mixed", "mixed"

    path = Path(voice)
    if path.suffix == ".pt" or path.exists():
        return "external", "local_file"

    return voice, "artifact"
