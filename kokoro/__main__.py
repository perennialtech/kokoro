"""TensorRT Kokoro TTS CLI."""

import argparse
import sys
import wave
from pathlib import Path
from typing import Generator, Union

import numpy as np
import torch
from loguru import logger

from kokoro import KokoroTRT
from kokoro.pipeline import LANGUAGE_CODES, infer_language_from_voice


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


def generate_audio(
    artifact_dir: Path,
    text: str,
    language: str,
    voice: Union[str, torch.Tensor],
    speed: float,
) -> Generator[torch.Tensor, None, None]:
    tts = KokoroTRT(artifact_dir)

    if (
        isinstance(voice, str)
        and Path(voice).suffix != ".pt"
        and not voice.startswith(language)
    ):
        logger.warning(f"Voice {voice} is not made for language {language}")

    for result in tts.synthesize(
        text=text,
        voice=voice,
        language=language,
        speed=speed,
        split_pattern=r"\n+",
    ):
        yield result.audio.detach().cpu().reshape(-1)


def generate_and_save_audio(
    output_file: Path,
    artifact_dir: Path,
    text: str,
    language: str,
    voice: str,
    speed: float,
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        for audio in generate_audio(
            artifact_dir=artifact_dir,
            text=text,
            language=language,
            voice=voice,
            speed=speed,
        ):
            audio_bytes = (
                (audio.numpy() * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
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
    args = parser.parse_args()

    configure_cli_logging(args.debug)

    if args.text is not None and args.input_file is not None:
        raise ValueError("You cannot specify both --text and --input-file")

    if args.text is not None:
        text = args.text
    elif args.input_file:
        text = args.input_file.read_text()
    else:
        print("Press Ctrl+D to stop reading input and start generating", flush=True)
        text = "".join(sys.stdin)

    language = infer_language_from_voice(args.language, args.voice)

    if args.output_file.suffix != ".wav":
        logger.warning("The output file name should end with .wav")

    generate_and_save_audio(
        output_file=args.output_file,
        artifact_dir=args.artifact_dir,
        text=text,
        language=language,
        voice=args.voice,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
