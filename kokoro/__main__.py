"""Kokoro TTS CLI"""

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

import numpy as np
import torch
from loguru import logger

languages = ["a", "b", "h", "e", "f", "i", "p", "j", "z"]
backends = ["pytorch", "onnx", "tensorrt"]


@dataclass
class BackendBundle:
    repo_id: str
    vocab: Optional[dict[str, int]]
    context_length: int
    backend: Any


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


def load_backend_bundle(
    backend: str,
    repo_id: Optional[str],
    config: Optional[Path],
    onnx_model_dir: Optional[Path],
    onnx_providers: Optional[Sequence[str]],
    pytorch_device: str,
    trt_artifact_dir: Optional[Path],
    trt_max_synthesis_frames: Optional[int],
    trt_fallback_to_pytorch: bool,
) -> BackendBundle:
    if backend == "pytorch":
        from kokoro import KModel

        model = KModel(repo_id=repo_id, config=config).eval()

        if pytorch_device == "auto":
            if torch.cuda.is_available():
                model = model.to("cuda")
        elif pytorch_device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "PyTorch CUDA device requested, but CUDA is not available"
                )
            model = model.to("cuda")
        elif pytorch_device == "cpu":
            model = model.to("cpu")
        else:
            raise ValueError(f"Unsupported PyTorch device: {pytorch_device}")

        return BackendBundle(
            repo_id=model.repo_id,
            vocab=model.vocab,
            context_length=model.context_length,
            backend=model.inference_backend(),
        )

    if backend == "onnx":
        from kokoro import KONNXModel

        if onnx_model_dir is None:
            raise ValueError("--onnx-model-dir is required when backend='onnx'")

        model = KONNXModel(
            onnx_model_dir,
            providers=onnx_providers,
        )
        return BackendBundle(
            repo_id=model.repo_id,
            vocab=model.vocab,
            context_length=model.context_length,
            backend=model.inference_backend(),
        )

    if backend == "tensorrt":
        from kokoro import KModel, KokoroTRTBackend

        if trt_artifact_dir is None:
            raise ValueError("--trt-artifact-dir is required when backend='tensorrt'")
        if not torch.cuda.is_available():
            raise ValueError("TensorRT backend requires CUDA")

        model = KModel(repo_id=repo_id, config=config).eval().to("cuda")
        inference_backend = KokoroTRTBackend(
            model,
            artifact_dir=trt_artifact_dir,
            max_synthesis_frames=trt_max_synthesis_frames,
            fallback_to_pytorch=trt_fallback_to_pytorch,
        )
        return BackendBundle(
            repo_id=model.repo_id,
            vocab=model.vocab,
            context_length=model.context_length,
            backend=inference_backend,
        )

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
    trt_artifact_dir: Optional[Path] = None,
    trt_max_synthesis_frames: Optional[int] = None,
    trt_fallback_to_pytorch: bool = True,
) -> Generator[torch.Tensor, None, None]:
    from kokoro import KPipeline

    bundle = load_backend_bundle(
        backend=backend,
        repo_id=repo_id,
        config=config,
        onnx_model_dir=onnx_model_dir,
        onnx_providers=onnx_providers,
        pytorch_device=pytorch_device,
        trt_artifact_dir=trt_artifact_dir,
        trt_max_synthesis_frames=trt_max_synthesis_frames,
        trt_fallback_to_pytorch=trt_fallback_to_pytorch,
    )

    frontend = KPipeline(
        lang_code=kokoro_language,
        repo_id=bundle.repo_id,
        vocab=bundle.vocab,
        context_length=bundle.context_length,
    )

    preferred_ref_device = getattr(bundle.backend, "preferred_ref_device", None)
    preferred_ref_dtype = getattr(bundle.backend, "preferred_ref_dtype", None)
    if preferred_ref_device is not None:
        frontend.set_voice_target(
            device=preferred_ref_device,
            dtype=preferred_ref_dtype or torch.float32,
        )

    if isinstance(voice, str) and not voice.startswith(kokoro_language):
        logger.warning(f"Voice {voice} is not made for language {kokoro_language}")

    for prepared in frontend.prepare(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r"\n+",
    ):
        logger.debug(prepared.phonemes)
        output = bundle.backend(prepared=prepared)
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
    trt_artifact_dir: Optional[Path] = None,
    trt_max_synthesis_frames: Optional[int] = None,
    trt_fallback_to_pytorch: bool = True,
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
            trt_artifact_dir=trt_artifact_dir,
            trt_max_synthesis_frames=trt_max_synthesis_frames,
            trt_fallback_to_pytorch=trt_fallback_to_pytorch,
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
        "-i",
        "--input-file",
        "--input_file",
        type=Path,
        help="Path to input text file",
    )
    parser.add_argument(
        "-t", "--text", help="Text to use instead of reading from stdin"
    )
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument(
        "--repo-id",
        "--repo_id",
        help="Hugging Face model repo for PyTorch/TensorRT config, weights, and voices",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a Kokoro config.json file for PyTorch/TensorRT",
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
        help="Directory containing self-contained ONNX export artifacts",
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
        "--trt-artifact-dir",
        "--trt_artifact_dir",
        type=Path,
        help="Directory containing compiled TensorRT decoder/generator artifacts",
    )
    parser.add_argument(
        "--trt-max-synthesis-frames",
        "--trt_max_synthesis_frames",
        type=int,
        help="Override TensorRT maximum synthesis-frame limit from artifact metadata",
    )
    parser.add_argument(
        "--no-trt-pytorch-fallback",
        "--no_trt_pytorch_fallback",
        action="store_true",
        help="Raise instead of falling back to PyTorch when TensorRT max frame limit is exceeded",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )
    args = parser.parse_args()
    configure_cli_logging(args.debug)

    onnx_providers = parse_onnx_providers(args.onnx_providers)
    if args.backend == "onnx" and args.onnx_model_dir is None:
        parser.error("--onnx-model-dir is required when --backend=onnx")
    if args.backend == "tensorrt" and args.trt_artifact_dir is None:
        parser.error("--trt-artifact-dir is required when --backend=tensorrt")

    lang = args.language or args.voice[0]

    if args.text is not None and args.input_file is not None:
        raise ValueError("You cannot specify both 'text' and 'input_file'")
    if args.text is not None:
        text = args.text
    elif args.input_file:
        text = args.input_file.read_text()
    else:
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
        trt_artifact_dir=args.trt_artifact_dir,
        trt_max_synthesis_frames=args.trt_max_synthesis_frames,
        trt_fallback_to_pytorch=not args.no_trt_pytorch_fallback,
    )


if __name__ == "__main__":
    main()
