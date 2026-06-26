from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, Union


class TraceSink(Protocol):
    def emit_trace(self, trace: Any) -> None: ...


def _clean(value: Any) -> Any:
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class JsonlTraceSink:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit_trace(self, trace: Any) -> None:
        with open(self.path, "a", encoding="utf-8") as w:
            json.dump(_clean(trace), w, sort_keys=True)
            w.write("\n")


class InMemoryTraceSink:
    def __init__(self):
        self.traces: list[Any] = []

    def emit_trace(self, trace: Any) -> None:
        self.traces.append(trace)


class LogSummarySink:
    def __init__(self, file: TextIO | None = None):
        self.file = file or sys.stderr

    def emit_trace(self, trace: Any) -> None:
        if getattr(trace, "kind", None) not in {"request", "chunk"}:
            return

        stages = getattr(trace, "stages", []) or []
        if not stages:
            return

        title = (
            f"{trace.kind} {getattr(trace, 'status', 'unknown')} "
            f"submit={getattr(trace, 'submit_latency_s', 0.0):.4f}s "
            f"ready={getattr(trace, 'ready_latency_s', None) or 0.0:.4f}s "
            f"rtf={getattr(trace, 'rtf_ready', None) or getattr(trace, 'rtf_submit', None) or 0.0:.4f}"
        )
        print(title, file=self.file)
        print(
            "stage                                      count  cpu_total_ms  cuda_total_ms",
            file=self.file,
        )

        totals: dict[str, tuple[int, float, float]] = {}
        for stage in stages:
            count, cpu, cuda = totals.get(stage.name, (0, 0.0, 0.0))
            totals[stage.name] = (
                count + 1,
                cpu + float(stage.cpu_ms),
                cuda + float(stage.cuda_ms or 0.0),
            )

        for name, (count, cpu, cuda) in sorted(totals.items()):
            print(f"{name:<42} {count:>5} {cpu:>12.3f} {cuda:>13.3f}", file=self.file)
        print("", file=self.file)
