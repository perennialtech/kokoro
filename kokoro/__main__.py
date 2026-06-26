"""TensorRT Kokoro TTS CLI."""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from loguru import logger

from kokoro import KokoroTRT
from kokoro.pipeline import LANGUAGE_CODES, infer_language_from_voice
from kokoro.telemetry import (JsonlTraceSink, LogSummarySink, ProfilerConfig,
                              PrometheusMetrics, Telemetry)


def configure_cli_logging(debug: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<cyan>{module:>16}:{line}</cyan> | "
            "<level>{level: >8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        level="DEBUG" if debug else "INFO",
    )
    logger.enable("kokoro")


def telemetry_from_args(args) -> Telemetry:
    sinks = []
    enabled = bool(args.profile or args.profile_jsonl)
    if args.profile:
        sinks.append(LogSummarySink())
    if args.profile_jsonl:
        sinks.append(JsonlTraceSink(args.profile_jsonl))

    metrics = None
    if args.metrics_prometheus_port:
        metrics = PrometheusMetrics(prefix=args.metrics_prefix)
        metrics.start_http_server(args.metrics_prometheus_port)

    return Telemetry(
        profiler_config=ProfilerConfig(
            enabled=enabled,
            cuda_timing=args.profile_cuda,
            synchronize_cuda=args.profile_sync_cuda,
            emit_nvtx=args.profile_nvtx,
            include_text=args.profile_include_text,
        ),
        trace_sinks=sinks,
        metrics=metrics,
    )


def read_input(args, profile) -> str:
    with profile.span("cli.read_input"):
        if args.text is not None and args.input_file is not None:
            raise ValueError("You cannot specify both --text and --input-file")
        if args.text is not None:
            return args.text
        if args.input_file:
            return args.input_file.read_text()
        print("Press Ctrl+D to stop reading input and start generating", flush=True)
        return "".join(sys.stdin)


def generate_and_save_audio(
    *,
    output_file: Path,
    tts: KokoroTRT,
    text: str,
    language: str,
    voice: str,
    speed: float,
    profile,
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        with profile.span("cli.synthesize"):
            prepared_items = list(
                tts.prepare(
                    text=text,
                    voice=voice,
                    language=language,
                    speed=speed,
                    split_pattern=r"\n+",
                    profile=profile,
                )
            )

            for prepared in prepared_items:
                result = tts.synthesize_prepared(prepared, profile=profile)
                with profile.span("cli.cpu_copy", cuda=True):
                    audio = result.audio.detach().cpu().reshape(-1)
                with profile.span("cli.wav_write"):
                    audio_bytes = (
                        (audio.numpy() * 32767)
                        .clip(-32768, 32767)
                        .astype(np.int16)
                        .tobytes()
                    )
                    wav_file.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kokoro TensorRT TTS")
    parser.add_argument(
        "--artifact-dir",
        "--artifact_dir",
        type=Path,
        required=True,
        help="TensorRT artifact directory",
    )
    parser.add_argument(
        "-m",
        "--voice",
        default="af_heart",
        help="Artifact-local voice name or .pt path",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=sorted(LANGUAGE_CODES),
        help="Language code",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        "--output_file",
        type=Path,
        required=True,
        help="Path to output WAV file",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        "--input_file",
        type=Path,
        help="Path to input text file",
    )
    parser.add_argument("-t", "--text", help="Text to use instead of reading stdin")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages")
    parser.add_argument(
        "--profile", action="store_true", help="Print profiling summary"
    )
    parser.add_argument(
        "--profile-jsonl", type=Path, help="Write JSONL profiling traces"
    )
    parser.add_argument(
        "--profile-cuda", action="store_true", help="Record CUDA events"
    )
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize once per chunk/request for ready latency",
    )
    parser.add_argument("--profile-nvtx", action="store_true", help="Emit NVTX ranges")
    parser.add_argument(
        "--profile-include-text",
        action="store_true",
        help="Include graphemes and phonemes in traces",
    )
    parser.add_argument(
        "--metrics-prometheus-port",
        type=int,
        help="Expose Prometheus metrics on this port",
    )
    parser.add_argument(
        "--metrics-prefix",
        default="kokoro_tts",
        help="Prometheus metric prefix",
    )
    args = parser.parse_args()

    configure_cli_logging(args.debug)
    telemetry = telemetry_from_args(args)
    language = infer_language_from_voice(args.language, args.voice)
    profile = telemetry.start_request(
        language=language,
        voice=args.voice,
        speed=args.speed,
        precision="",
    )

    status = "cancelled"
    error = None
    try:
        with profile.span("cli.total"):
            text = read_input(args, profile)
            profile.trace.input_chars = len(text)

            if args.output_file.suffix != ".wav":
                logger.warning("The output file name should end with .wav")

            tts = KokoroTRT(args.artifact_dir, telemetry=telemetry)

            if (
                isinstance(args.voice, str)
                and Path(args.voice).suffix != ".pt"
                and not args.voice.startswith(language)
            ):
                logger.warning(
                    f"Voice {args.voice} is not made for language {language}"
                )

            generate_and_save_audio(
                output_file=args.output_file,
                tts=tts,
                text=text,
                language=language,
                voice=args.voice,
                speed=args.speed,
                profile=profile,
            )
        status = "ok"
    except Exception as e:
        status = "error"
        error = e
        raise
    finally:
        profile.finalize(status, error)


if __name__ == "__main__":
    main()
