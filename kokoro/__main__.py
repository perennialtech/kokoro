"""Kokoro TTS CLI"""

import argparse
import wave
from pathlib import Path
from typing import Generator, TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

languages = ["a", "b", "h", "e", "f", "i", "p", "j", "z"]

if TYPE_CHECKING:
    pass


def generate_audio(
    text: str,
    kokoro_language: str,
    voice: str,
    speed=1,
) -> Generator[torch.Tensor, None, None]:
    from kokoro import KModel, KPipeline, KokoroInferenceBackend

    model = KModel().eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    frontend = KPipeline(
        lang_code=kokoro_language,
        repo_id=model.repo_id,
        vocab=model.vocab,
        context_length=model.context_length,
    )
    backend = KokoroInferenceBackend(model)

    if isinstance(voice, str) and not voice.startswith(kokoro_language):
        logger.warning(f"Voice {voice} is not made for language {kokoro_language}")

    for prepared in frontend.prepare(
        text, voice=voice, speed=speed, split_pattern=r"\n+"
    ):
        logger.debug(prepared.phonemes)
        output = backend(prepared=prepared)
        yield output.audio.detach().cpu().reshape(-1)


def generate_and_save_audio(
    output_file: Path,
    text: str,
    kokoro_language: str,
    voice: str,
    speed=1,
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        for audio in generate_audio(
            text, kokoro_language=kokoro_language, voice=voice, speed=speed
        ):
            audio_bytes = (
                (audio.numpy() * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            )
            wav_file.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--voice", default="af_heart", help="Voice to use")
    parser.add_argument("-l", "--language", choices=languages, help="Language to use")
    parser.add_argument(
        "-o",
        "--output-file",
        "--output_file",
        type=Path,
        required=True,
        help="Path to output WAV file",
    )
    parser.add_argument(
        "-i", "--input-file", "--input_file", type=Path, help="Path to input text file"
    )
    parser.add_argument(
        "-t", "--text", help="Text to use instead of reading from stdin"
    )
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    if args.debug:
        logger.enable("kokoro")

    lang = args.language or args.voice[0]

    if args.text is not None and args.input_file is not None:
        raise ValueError("You cannot specify both 'text' and 'input_file'")
    if args.text is not None:
        text = args.text
    elif args.input_file:
        text = args.input_file.read_text()
    else:
        import sys

        print("Press Ctrl+D to stop reading input and start generating", flush=True)
        text = "".join(sys.stdin)

    if args.output_file.suffix != ".wav":
        logger.warning("The output file name should end with .wav")

    generate_and_save_audio(
        output_file=args.output_file,
        text=text,
        kokoro_language=lang,
        voice=args.voice,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
