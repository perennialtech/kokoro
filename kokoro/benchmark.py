"""Measure first CPU-ready audio for shuffled, independently synthesized texts."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

import kokoro
from kokoro import KokoroTRT
from kokoro.compile import sha256_file

SHORT_TEXTS = (
    "Hi.",
    "Yes!",
    "No, thanks.",
    "Hello, world!",
    "Good morning.",
    "Are you there?",
    "How are you doing today?",
    "Please try again.",
    "That sounds good to me.",
    "I'll be back in five minutes.",
    "The train leaves at 9:30.",
    "Your total is $42.50.",
    "What's the weather like today?",
    "Please turn off the kitchen lights.",
    "Can you help me find my keys?",
    "Sorry, I didn't catch that.",
    "Thank you for your patience.",
    "Let's take a short break.",
    "There are three messages for you.",
    "The quick brown fox jumps over the lazy dog.",
    "One, two, three, four, five.",
    "Well... that was unexpected!",
    "Is this really the right address?",
    "Everything is ready. Shall we begin?",
    "Please turn off all the lights when you head home.",
)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "p99": ordered[round(0.99 * (len(ordered) - 1))],
    }


def gpu_processes() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    return {
        int(pid): int(memory) * 2**20
        for line in output.splitlines()
        for pid, memory in [line.split(",")]
    }


class ProcessMemorySampler:
    """Sample new GPU processes, excluding workloads present before model load."""

    def __init__(self):
        self.peak_bytes: int | None = None
        self.done = threading.Event()
        self.thread: threading.Thread | None = None
        try:
            self.existing = set(gpu_processes())
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        self.peak_bytes = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.done.is_set():
            try:
                processes = gpu_processes()
                used = sum(
                    value
                    for pid, value in processes.items()
                    if pid not in self.existing
                )
                self.peak_bytes = max(self.peak_bytes or 0, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                self.peak_bytes = None
                return
            self.done.wait(0.05)

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--input-file", type=Path, help="One independent request per nonempty line"
    )
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--default-han-language", choices=("zh", "ja"), default="zh")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--text-graph-max-tokens", type=int)
    parser.add_argument("--render-graph-max-frames", type=int)
    parser.add_argument(
        "--eval-host",
        action="store_true",
        help="Evaluate older runtimes without dropout",
    )
    parser.add_argument(
        "--tf32", action="store_true", help="Match the math mode enabled by Chatterbox"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0 or args.threads < 1:
        parser.error(
            "iterations and threads must be positive; warmups must be non-negative"
        )
    texts = (
        list(SHORT_TEXTS)
        if args.input_file is None
        else [text for text in args.input_file.read_text().splitlines() if text.strip()]
    )
    if not texts:
        parser.error("the corpus must contain at least one nonempty text")

    torch.set_num_threads(args.threads)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    sampler = ProcessMemorySampler()
    options = {}
    if args.text_graph_max_tokens is not None:
        options["text_graph_max_tokens"] = args.text_graph_max_tokens
    if args.render_graph_max_frames is not None:
        options["render_graph_max_frames"] = args.render_graph_max_frames
    try:
        start = time.perf_counter()
        tts = KokoroTRT(args.artifact_dir, **options)
        if args.eval_host:
            tts.host.eval()
        for _ in range(args.warmups):
            for result in tts.synthesize(
                "Warm up.", args.voice, args.default_han_language
            ):
                result.audio.cpu()
        torch.cuda.synchronize()
        startup_s = time.perf_counter() - start
        timings: list[list[float]] = [[] for _ in texts]
        first_seen: list[float | None] = [None for _ in texts]
        samples: list[int] = [0 for _ in texts]
        rng = random.Random(args.seed)
        for _ in range(args.iterations):
            order = list(range(len(texts)))
            rng.shuffle(order)
            for index in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                first_ms = None
                total_samples = 0
                for result in tts.synthesize(
                    texts[index],
                    args.voice,
                    args.default_han_language,
                    speed=args.speed,
                ):
                    audio = result.audio.cpu()
                    total_samples += audio.numel()
                    if first_ms is None and audio.numel():
                        first_ms = (time.perf_counter() - start) * 1000
                if first_ms is None:
                    raise RuntimeError(f"Corpus item {index} produced no audio")
                timings[index].append(first_ms)
                if first_seen[index] is None:
                    first_seen[index] = first_ms
                samples[index] = total_samples
        torch.cuda.synchronize()
        summary = {
            "ttfa_ms": distribution([value for row in timings for value in row]),
            "startup_and_warmup_s": startup_s,
            "requests": len(texts) * args.iterations,
            "voice": args.voice,
            "speed": args.speed,
            "precision": tts.metadata.precision,
            "engine_sha256": sha256_file(str(tts.artifact.paths.engine_path)),
            "torch_version": torch.__version__,
            "runtime_source": kokoro.__file__,
            "tf32": args.tf32,
            "gpu": torch.cuda.get_device_name(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cases": [
                {
                    "index": i,
                    "chars": len(text),
                    "first_seen_ms": first_seen[i],
                    "ttfa_ms": distribution(timings[i]),
                    "samples": samples[i],
                }
                for i, text in enumerate(texts)
            ],
        }
    finally:
        sampler.close()
    summary["sampled_new_gpu_process_peak_bytes"] = sampler.peak_bytes
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
