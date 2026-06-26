import pytest
import torch

from kokoro.telemetry import InMemoryMetrics, ProfilerConfig, Telemetry


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA telemetry requires CUDA"
)
def test_cuda_event_timing_records_nonnegative_cuda_ms():
    telemetry = Telemetry(
        ProfilerConfig(enabled=True, cuda_timing=True, synchronize_cuda=True)
    )
    chunk = telemetry.start_chunk()
    with chunk.span("cuda.work", cuda=True):
        x = torch.ones(1024, device="cuda")
        y = x * 2
        del y
    chunk.finalize("ok")

    stage = chunk.trace.stages[0]
    assert stage.cuda_ms is not None
    assert stage.cuda_ms >= 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA telemetry requires CUDA"
)
def test_synchronized_chunk_ready_latency_and_memory_gauges():
    metrics = InMemoryMetrics()
    telemetry = Telemetry(
        ProfilerConfig(enabled=True, cuda_timing=True, synchronize_cuda=True),
        metrics=metrics,
    )
    chunk = telemetry.start_chunk()
    with chunk.span("cuda.work", cuda=True):
        torch.empty(1024, device="cuda")
    chunk.finalize("ok")

    assert chunk.trace.ready_latency_s is not None
    assert any(key[0] == "cuda_memory_allocated_bytes" for key in metrics.gauges)
    assert any(key[0] == "cuda_memory_reserved_bytes" for key in metrics.gauges)
    assert any(key[0] == "cuda_max_memory_allocated_bytes" for key in metrics.gauges)
