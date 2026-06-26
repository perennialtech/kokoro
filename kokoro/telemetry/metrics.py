from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Protocol

Labels = Mapping[str, str]


class MetricsSink(Protocol):
    def observe_counter(
        self, name: str, value: float = 1, labels: Labels = {}
    ) -> None: ...

    def observe_histogram(
        self, name: str, value: float, labels: Labels = {}
    ) -> None: ...

    def set_gauge(self, name: str, value: float, labels: Labels = {}) -> None: ...

    def set_info(self, name: str, labels: Labels) -> None: ...


class NoOpMetrics:
    def observe_counter(self, name: str, value: float = 1, labels: Labels = {}) -> None:
        return

    def observe_histogram(self, name: str, value: float, labels: Labels = {}) -> None:
        return

    def set_gauge(self, name: str, value: float, labels: Labels = {}) -> None:
        return

    def set_info(self, name: str, labels: Labels) -> None:
        return


class InMemoryMetrics:
    def __init__(self):
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = (
            defaultdict(float)
        )
        self.histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = (
            defaultdict(list)
        )
        self.gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self.infos: dict[str, dict[str, str]] = {}

    @staticmethod
    def _key(name: str, labels: Labels) -> tuple[str, tuple[tuple[str, str], ...]]:
        return name, tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    def observe_counter(self, name: str, value: float = 1, labels: Labels = {}) -> None:
        self.counters[self._key(name, labels)] += float(value)

    def observe_histogram(self, name: str, value: float, labels: Labels = {}) -> None:
        self.histograms[self._key(name, labels)].append(float(value))

    def set_gauge(self, name: str, value: float, labels: Labels = {}) -> None:
        self.gauges[self._key(name, labels)] = float(value)

    def set_info(self, name: str, labels: Labels) -> None:
        self.infos[name] = {str(k): str(v) for k, v in labels.items()}


class PrometheusMetrics:
    STAGE_BUCKETS = (
        0.0005,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
    )
    REQUEST_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
    RTF_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5)
    FRAME_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

    def __init__(self, prefix: str = "kokoro_tts", registry=None):
        from prometheus_client import CollectorRegistry

        self.prefix = prefix.strip("_")
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self._metrics: dict[tuple[str, str], object] = {}

    def start_http_server(self, port: int, addr: str = "0.0.0.0") -> None:
        from prometheus_client import start_http_server

        start_http_server(int(port), addr=addr, registry=self.registry)

    def _full_name(self, name: str) -> str:
        return name if name.startswith(f"{self.prefix}_") else f"{self.prefix}_{name}"

    def _metric(self, kind: str, name: str, labels: Labels):
        key = kind, name
        existing = self._metrics.get(key)
        if existing is not None:
            return existing

        labelnames = tuple(labels.keys())
        full_name = self._full_name(name)
        description = full_name.replace("_", " ")

        if kind == "counter":
            from prometheus_client import Counter

            metric = Counter(full_name, description, labelnames, registry=self.registry)
        elif kind == "histogram":
            from prometheus_client import Histogram

            buckets = self._buckets_for(name)
            metric = Histogram(
                full_name,
                description,
                labelnames,
                buckets=buckets,
                registry=self.registry,
            )
        elif kind == "gauge":
            from prometheus_client import Gauge

            metric = Gauge(full_name, description, labelnames, registry=self.registry)
        elif kind == "info":
            from prometheus_client import Info

            metric = Info(full_name, description, registry=self.registry)
        else:
            raise ValueError(f"Unsupported Prometheus metric kind: {kind}")

        self._metrics[key] = metric
        return metric

    def _buckets_for(self, name: str):
        if "rtf" in name:
            return self.RTF_BUCKETS
        if "frames" in name or "sample_length" in name or "input_ids" in name:
            return self.FRAME_BUCKETS
        if "request" in name or "chunk" in name:
            return self.REQUEST_BUCKETS
        return self.STAGE_BUCKETS

    def observe_counter(self, name: str, value: float = 1, labels: Labels = {}) -> None:
        metric = self._metric("counter", name, labels)
        if labels:
            metric.labels(**dict(labels)).inc(float(value))
        else:
            metric.inc(float(value))

    def observe_histogram(self, name: str, value: float, labels: Labels = {}) -> None:
        metric = self._metric("histogram", name, labels)
        if labels:
            metric.labels(**dict(labels)).observe(float(value))
        else:
            metric.observe(float(value))

    def set_gauge(self, name: str, value: float, labels: Labels = {}) -> None:
        metric = self._metric("gauge", name, labels)
        if labels:
            metric.labels(**dict(labels)).set(float(value))
        else:
            metric.set(float(value))

    def set_info(self, name: str, labels: Labels) -> None:
        metric = self._metric("info", name, {})
        metric.info({str(k): str(v) for k, v in labels.items()})
