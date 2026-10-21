from unittest.mock import MagicMock

import kokoro.benchmark as benchmark
from kokoro.benchmark import SHORT_TEXTS, ProcessMemorySampler, distribution


def test_default_corpus_is_short_and_varied():
    assert all(0 < len(text) <= 50 for text in SHORT_TEXTS)
    assert max(map(len, SHORT_TEXTS)) == 50
    assert len(set(SHORT_TEXTS)) == len(SHORT_TEXTS)
    assert len(set(map(len, SHORT_TEXTS))) > 10


def test_latency_distribution():
    assert distribution([4, 1, 3, 2]) == {"p50": 2.5, "p95": 4, "p99": 4}


def test_memory_sampler_excludes_existing_processes(monkeypatch):
    sampler = ProcessMemorySampler.__new__(ProcessMemorySampler)
    sampler.existing = {7}
    sampler.peak_bytes = 0
    sampler.done = MagicMock()
    sampler.done.is_set.side_effect = [False, True]
    monkeypatch.setattr(benchmark, "gpu_processes", lambda: {7: 4096, 8: 1024})
    sampler._sample()
    assert sampler.peak_bytes == 1024


def test_memory_sampler_reports_unavailable_on_error(monkeypatch):
    sampler = ProcessMemorySampler.__new__(ProcessMemorySampler)
    sampler.peak_bytes = 0
    sampler.done = MagicMock()
    sampler.done.is_set.return_value = False

    def unavailable():
        raise OSError("nvidia-smi unavailable")

    monkeypatch.setattr(benchmark, "gpu_processes", unavailable)
    sampler._sample()
    assert sampler.peak_bytes is None
