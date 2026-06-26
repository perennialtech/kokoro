from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

from kokoro import KokoroTRT
from kokoro.pipeline import LANGUAGE_CODES, infer_language_from_voice
from kokoro.telemetry import JsonlTraceSink, ProfilerConfig, Telemetry


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarize(traces, chars: int) -> dict:
    request_latencies = [
        (t.ready_latency_s or t.submit_latency_s) for t in traces if t.status == "ok"
    ]
    chunk_latencies = [
        (c.ready_latency_s or c.submit_latency_s)
        for t in traces
        for c in t.chunks
        if c.status == "ok"
    ]
    rtfs = [
        (c.rtf_ready or c.rtf_submit)
        for t in traces
        for c in t.chunks
        if c.status == "ok" and (c.rtf_ready is not None or c.rtf_submit is not None)
    ]
    audio_s = sum(t.audio_duration_s for t in traces if t.status == "ok")
    wall_s = sum(request_latencies)
    stage_cpu = defaultdict(float)
    stage_cuda = defaultdict(float)
    errors = Counter()

    for trace in traces:
        if trace.error_type:
            errors[trace.error_type] += 1
        for stage in trace.stages:
            stage_cpu[stage.name] += stage.cpu_ms / 1000.0
            if stage.cuda_ms is not None:
                stage_cuda[stage.name] += stage.cuda_ms / 1000.0
            if stage.error_type:
                errors[f"{stage.name}:{stage.error_type}"] += 1
        for chunk in trace.chunks:
            if chunk.error_type:
                errors[chunk.error_type] += 1
            for stage in chunk.stages:
                stage_cpu[stage.name] += stage.cpu_ms / 1000.0
                if stage.cuda_ms is not None:
                    stage_cuda[stage.name] += stage.cuda_ms / 1000.0
                if stage.error_type:
                    errors[f"{stage.name}:{stage.error_type}"] += 1

    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    return {
        "request_ready_latency_s": {
            "p50": percentile(request_latencies, 50),
            "p90": percentile(request_latencies, 90),
            "p95": percentile(request_latencies, 95),
            "p99": percentile(request_latencies, 99),
        },
        "chunk_ready_latency_s": {
            "p50": percentile(chunk_latencies, 50),
            "p90": percentile(chunk_latencies, 90),
            "p95": percentile(chunk_latencies, 95),
            "p99": percentile(chunk_latencies, 99),
        },
        "rtf": {
            "p50": percentile(rtfs, 50),
            "p95": percentile(rtfs, 95),
            "mean": statistics.mean(rtfs) if rtfs else 0.0,
        },
        "audio_seconds": audio_s,
        "wall_seconds": wall_s,
        "audio_seconds_per_wall_second": audio_s / wall_s if wall_s > 0 else 0.0,
        "chars_per_second": chars / wall_s if wall_s > 0 else 0.0,
        "stage_cpu_seconds": dict(sorted(stage_cpu.items())),
        "stage_cuda_seconds": dict(sorted(stage_cuda.items())),
        "peak_gpu_memory_bytes": int(peak_gpu),
        "errors_by_stage": dict(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Kokoro TensorRT runtime")
    parser.add_argument("--artifact-dir", "--artifact_dir", type=Path, required=True)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--language", choices=sorted(LANGUAGE_CODES))
    parser.add_argument("--text")
    parser.add_argument("--input-file", "--input_file", type=Path)
    parser.add_argument("--split-pattern", "--split_pattern", default=r"\n+")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json", type=Path, help="Write benchmark summary JSON")
    parser.add_argument("--jsonl", type=Path, help="Write request/chunk traces JSONL")
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--profile-sync-cuda", action="store_true")
    parser.add_argument("--profile-nvtx", action="store_true")
    args = parser.parse_args()

    if args.text is not None and args.input_file is not None:
        raise ValueError("Use --text or --input-file, not both")
    text = args.text if args.text is not None else args.input_file.read_text()
    language = infer_language_from_voice(args.language, args.voice)

    sinks = [JsonlTraceSink(args.jsonl)] if args.jsonl else []
    telemetry = Telemetry(
        profiler_config=ProfilerConfig(
            enabled=True,
            cuda_timing=args.profile_cuda,
            synchronize_cuda=args.profile_sync_cuda,
            emit_nvtx=args.profile_nvtx,
        ),
        trace_sinks=sinks,
    )
    tts = KokoroTRT(args.artifact_dir, telemetry=telemetry)

    for _ in range(args.warmups):
        list(
            tts.synthesize(
                text=text,
                voice=args.voice,
                language=language,
                speed=args.speed,
                split_pattern=args.split_pattern,
            )
        )

    traces = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.iterations):
        try:
            list(
                tts.synthesize(
                    text=text,
                    voice=args.voice,
                    language=language,
                    speed=args.speed,
                    split_pattern=args.split_pattern,
                )
            )
        finally:
            if telemetry.last_request_trace is not None:
                traces.append(telemetry.last_request_trace)

    summary = summarize(traces, chars=len(text) * args.iterations)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
