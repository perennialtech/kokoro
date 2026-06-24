"""Kokoro TTS CLI"""

import argparse
import wave
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

import numpy as np
import torch
from loguru import logger

languages = ["a", "b", "h", "e", "f", "i", "p", "j", "z"]
backends = ["pytorch", "onnx"]


def parse_onnx_providers(provider_args: Optional[list[str]]) -> Optional[list[str]]:
    if not provider_args:
        return None

    providers: list[str] = []
    for provider_group in provider_args:
        providers.extend(
            provider.strip()
            for provider in provider_group.split(",")
            if provider.strip()
        )

    return providers or None


def load_model_and_backend(
    backend: str,
    repo_id: Optional[str],
    config: Optional[Path],
    onnx_model_dir: Optional[Path],
    onnx_providers: Optional[Sequence[str]],
    pytorch_device: str,
) -> tuple[Any, Any]:
    if backend == "pytorch":
        from kokoro import KModel

        model = KModel(repo_id=repo_id, config=config).eval()

        if pytorch_device == "auto":
            if torch.cuda.is_available():
                model = model.to("cuda")
        elif pytorch_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("PyTorch CUDA device requested, but CUDA is not available")
            model = model.to("cuda")
        elif pytorch_device == "cpu":
            model = model.to("cpu")
        else:
            raise ValueError(f"Unsupported PyTorch device: {pytorch_device}")

        return model, model.inference_backend()

    if backend == "onnx":
        from kokoro import KONNXModel

        if onnx_model_dir is None:
            raise ValueError("--onnx-model-dir is required when backend='onnx'")

        model = KONNXModel(
            onnx_model_dir,
            repo_id=repo_id,
            config=config,
            providers=onnx_providers,
        )
        return model, model.inference_backend()

    raise ValueError(f"Unsupported backend: {backend}")


def generate_audio(
    text: str,
    kokoro_language: str,
    voice: str,
    speed=1,
    backend: str = "pytorch",
    repo_id: Optional[str] = None,
    config: Optional[Path] = None,
    onnx_model_dir: Optional[Path] = None,
    onnx_providers: Optional[Sequence[str]] = None,
    pytorch_device: str = "auto",
) -> Generator[torch.Tensor, None, None]:
    from kokoro import KPipeline

    model, inference_backend = load_model_and_backend(
        backend=backend,
        repo_id=repo_id,
        config=config,
        onnx_model_dir=onnx_model_dir,
        onnx_providers=onnx_providers,
        pytorch_device=pytorch_device,
    )

    frontend = KPipeline(
        lang_code=kokoro_language,
        repo_id=model.repo_id,
        vocab=model.vocab,
        context_length=model.context_length,
    )

    if isinstance(voice, str) and not voice.startswith(kokoro_language):
        logger.warning(f"Voice {voice} is not made for language {kokoro_language}")

    for prepared in frontend.prepare(
        text, voice=voice, speed=speed, split_pattern=r"\n+"
    ):
        logger.debug(prepared.phonemes)
        output = inference_backend(prepared=prepared)
        for utterance in output.utterances:
            yield utterance.audio.detach().cpu().reshape(-1)


def generate_and_save_audio(
    output_file: Path,
    text: str,
    kokoro_language: str,
    voice: str,
    speed=1,
    backend: str = "pytorch",
    repo_id: Optional[str] = None,
    config: Optional[Path] = None,
    onnx_model_dir: Optional[Path] = None,
    onnx_providers: Optional[Sequence[str]] = None,
    pytorch_device: str = "auto",
) -> None:
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        for audio in generate_audio(
            text,
            kokoro_language=kokoro_language,
            voice=voice,
            speed=speed,
            backend=backend,
            repo_id=repo_id,
            config=config,
            onnx_model_dir=onnx_model_dir,
            onnx_providers=onnx_providers,
            pytorch_device=pytorch_device,
        ):
            audio_bytes = (
                (audio.numpy() * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            )
            wav_file.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=backends,
        default="pytorch",
        help="Inference backend to use",
    )
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
        "--repo-id",
        "--repo_id",
        help="Hugging Face model repo for config, PyTorch weights, and voices",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a Kokoro config.json file",
    )
    parser.add_argument(
        "--pytorch-device",
        "--pytorch_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for the PyTorch backend",
    )
    parser.add_argument(
        "--onnx-model-dir",
        "--onnx_model_dir",
        type=Path,
        help="Directory containing text_duration.onnx and acoustic_vocoder.onnx",
    )
    parser.add_argument(
        "--onnx-provider",
        "--onnx_provider",
        action="append",
        dest="onnx_providers",
        help=(
            "ONNX Runtime execution provider. Repeat this option or pass a "
            "comma-separated list, e.g. CUDAExecutionProvider,CPUExecutionProvider."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    if args.debug:
        logger.enable("kokoro")

    onnx_providers = parse_onnx_providers(args.onnx_providers)
    if args.backend == "onnx" and args.onnx_model_dir is None:
        parser.error("--onnx-model-dir is required when --backend=onnx")

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
        backend=args.backend,
        repo_id=args.repo_id,
        config=args.config,
        onnx_model_dir=args.onnx_model_dir,
        onnx_providers=onnx_providers,
        pytorch_device=args.pytorch_device,
    )


if __name__ == "__main__":
    main()
