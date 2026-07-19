import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional, Union

import torch
from misaki import MultilingualG2P, espeak

from .telemetry import (NoOpProfileContext, ProfileContext,
                        normalize_voice_label)


@dataclass
class PreparedInput:
    graphemes: str
    phonemes: str
    input_ids: torch.Tensor
    ref_s: torch.Tensor
    speed: torch.Tensor
    text_index: Optional[int] = None

    @property
    def input_length(self) -> int:
        return int(self.input_ids.shape[1])


class VoiceStore:
    def __init__(self, voice_dir: Union[str, Path]):
        self.voice_dir = Path(voice_dir)
        self.cpu_cache: dict[str, torch.Tensor] = {}
        self.device_cache: dict[tuple[str, str, str], torch.Tensor] = {}
        self.target_device: Optional[torch.device] = None
        self.target_dtype: Optional[torch.dtype] = None

    def set_target(
        self,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.target_device = torch.device(device)
        self.target_dtype = dtype

    @staticmethod
    def _validate_american_voice(voice: str) -> None:
        name = Path(voice).stem
        if not name.startswith("a"):
            raise ValueError(
                f"Only American English voices are supported, not {name!r}"
            )

    def _voice_path(self, voice: str) -> Path:
        self._validate_american_voice(voice)

        path = Path(voice)
        if path.exists():
            return path
        if path.suffix == ".pt":
            raise FileNotFoundError(f"Voice file does not exist: {path}")

        artifact_path = self.voice_dir / f"{voice}.pt"
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Voice {voice!r} was not found in artifact voice directory "
                f"{self.voice_dir}. Recompile the artifact with --include-voice "
                "or pass a local .pt voice path."
            )
        return artifact_path

    @staticmethod
    def _normalize_pack(pack: torch.Tensor) -> torch.Tensor:
        if pack.dim() == 1:
            return pack.unsqueeze(0)
        if pack.dim() > 2:
            return pack.view(-1, pack.shape[-1])
        return pack

    def load_single_voice(
        self,
        voice: str,
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        self._validate_american_voice(voice)
        voice_label, voice_kind = normalize_voice_label(voice)

        if voice in self.cpu_cache:
            with profile.span(
                "voice.cache_cpu_hit",
                attrs={"voice_kind": voice_kind, "voice_label": voice_label},
            ):
                profile.counter(
                    "voice_cache_events_total",
                    1,
                    {"cache": "cpu", "result": "hit", "voice_kind": voice_kind},
                )
            return self.cpu_cache[voice]

        profile.counter(
            "voice_cache_events_total",
            1,
            {"cache": "cpu", "result": "miss", "voice_kind": voice_kind},
        )

        with profile.span(
            "voice.resolve_path",
            attrs={"voice_kind": voice_kind, "voice_label": voice_label},
        ):
            path = self._voice_path(voice)

        with profile.span("voice.load_cpu", attrs={"voice_kind": voice_kind}) as span:
            pack = self._normalize_pack(torch.load(path, weights_only=True))
            span.attrs["pack_rows"] = int(pack.shape[0])
            span.attrs["pack_dim"] = int(pack.shape[-1])
            span.attrs["bytes"] = int(pack.numel() * pack.element_size())
            profile.counter(
                "voice_loads_total",
                1,
                {"voice_kind": voice_kind, "result": "ok"},
            )
        self.cpu_cache[voice] = pack
        return pack

    def _load_cpu(
        self,
        voice: str,
        *,
        delimiter: str = ",",
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        voice_label, voice_kind = normalize_voice_label(voice)

        if voice in self.cpu_cache:
            with profile.span(
                "voice.cache_cpu_hit",
                attrs={"voice_kind": voice_kind, "voice_label": voice_label},
            ):
                profile.counter(
                    "voice_cache_events_total",
                    1,
                    {"cache": "cpu", "result": "hit", "voice_kind": voice_kind},
                )
            return self.cpu_cache[voice]

        packs = [
            self.load_single_voice(v.strip(), profile=profile)
            for v in voice.split(delimiter)
            if v.strip()
        ]
        if not packs:
            raise ValueError("voice must not be empty")

        if len(packs) == 1:
            self.cpu_cache[voice] = packs[0]
        else:
            with profile.span("voice.blend", attrs={"voice_kind": "mixed"}) as span:
                self.cpu_cache[voice] = torch.mean(torch.stack(packs), dim=0)
                span.attrs["voices"] = len(packs)
                profile.counter(
                    "voice_loads_total",
                    1,
                    {"voice_kind": "mixed", "result": "blend"},
                )
        return self.cpu_cache[voice]

    def load(
        self,
        voice: str,
        *,
        delimiter: str = ",",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        voice_label, voice_kind = normalize_voice_label(voice)
        base = self._load_cpu(
            voice,
            delimiter=delimiter,
            profile=profile,
        )

        target_device = (
            torch.device(device) if device is not None else self.target_device
        )
        target_dtype = dtype or self.target_dtype

        if target_device is None and target_dtype is None:
            return base

        if target_device is None:
            target_device = base.device
        if target_dtype is None:
            target_dtype = base.dtype

        key = (voice, str(target_device), str(target_dtype))
        cached = self.device_cache.get(key)
        if cached is not None:
            with profile.span(
                "voice.cache_device_hit",
                attrs={"voice_kind": voice_kind, "voice_label": voice_label},
            ):
                profile.counter(
                    "voice_cache_events_total",
                    1,
                    {"cache": "device", "result": "hit", "voice_kind": voice_kind},
                )
            return cached

        profile.counter(
            "voice_cache_events_total",
            1,
            {"cache": "device", "result": "miss", "voice_kind": voice_kind},
        )
        with profile.span(
            "voice.transfer_device",
            cuda=target_device.type == "cuda",
            attrs={"voice_kind": voice_kind, "device": str(target_device)},
        ) as span:
            cached = base.to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=target_device.type == "cuda",
            )
            span.attrs["bytes"] = int(cached.numel() * cached.element_size())
        self.device_cache[key] = cached
        return cached


class TextFrontend:
    def __init__(
        self,
        default_han_language: Literal["zh", "ja"],
        vocab: dict[str, int],
        context_length: int,
        voice_store: VoiceStore,
    ):
        if context_length <= 2:
            raise ValueError(
                f"context_length must be greater than 2, got {context_length}"
            )

        self.vocab = vocab
        self.context_length = int(context_length)
        self.max_phoneme_len = self.context_length - 2
        self.voice_store = voice_store
        self.g2p = MultilingualG2P(
            default_han_language=default_han_language,
            fallback=espeak.EspeakFallback(british=False),
        )

    def phonemes_to_ids(self, phonemes: str) -> List[int]:
        return [self.vocab[p] for p in phonemes if self.vocab.get(p) is not None]

    def phoneme_id_count(self, phonemes: str) -> int:
        return sum(1 for p in phonemes if self.vocab.get(p) is not None)

    def split_phonemes_to_context(self, phonemes: str) -> Generator[str, None, None]:
        current: list[str] = []
        count = 0

        for p in phonemes:
            increment = 1 if self.vocab.get(p) is not None else 0
            if count + increment > self.max_phoneme_len and current:
                chunk = "".join(current).strip()
                if chunk:
                    yield chunk
                current = []
                count = 0

            current.append(p)
            count += increment

        chunk = "".join(current).strip()
        if chunk:
            yield chunk

    def prepare_phonemes(
        self,
        graphemes: str,
        phonemes: str,
        voice: str,
        speed: Union[float, Callable[[int], float]] = 1,
        text_index: Optional[int] = None,
        profile: Optional[ProfileContext] = None,
    ) -> PreparedInput:
        profile = profile or NoOpProfileContext()
        with profile.span("frontend.prepare_phonemes") as span:
            if not phonemes:
                raise ValueError("Cannot prepare empty phoneme string")

            ids = self.phonemes_to_ids(phonemes)
            if not ids:
                raise ValueError(
                    "Phoneme string does not contain any symbols in the model vocabulary"
                )
            if len(ids) > self.max_phoneme_len:
                raise ValueError(
                    f"Tokenized phoneme payload too long: {len(ids)} > "
                    f"{self.max_phoneme_len}"
                )

            pack = self.voice_store.load(
                voice,
                profile=profile,
            )
            ref_index = min(len(phonemes) - 1, pack.shape[0] - 1)
            speed_value = speed(len(phonemes)) if callable(speed) else speed
            span.attrs.update(
                {
                    "input_chars": len(graphemes),
                    "phoneme_chars": len(phonemes),
                    "phoneme_ids": len(ids),
                    "ref_index": int(ref_index),
                }
            )

            return PreparedInput(
                graphemes=graphemes,
                phonemes=phonemes,
                text_index=text_index,
                input_ids=torch.tensor([[0, *ids, 0]], dtype=torch.long),
                ref_s=pack[ref_index].reshape(1, -1).contiguous(),
                speed=torch.tensor(
                    [float(speed_value)],
                    dtype=torch.float32,
                    device=pack.device,
                ),
            )

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
        chunks: list[str] = []
        current = ""

        for sentence in re.split(r"(?<=[.!?。！？])", text):
            while sentence:
                remaining = chunk_size - len(current)
                if len(sentence) <= remaining:
                    current += sentence
                    break

                if current:
                    chunk = current.strip()
                    if chunk:
                        chunks.append(chunk)
                    current = ""
                    continue

                chunk = sentence[:chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                sentence = sentence[chunk_size:]

        chunk = current.strip()
        if chunk:
            chunks.append(chunk)

        return chunks

    def prepare(
        self,
        text: Union[str, List[str]],
        voice: str,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r"\n+",
        profile: Optional[ProfileContext] = None,
    ) -> Generator[PreparedInput, None, None]:
        profile = profile or NoOpProfileContext()
        prepared_items: list[PreparedInput] = []

        with profile.span("frontend.prepare_total") as total_span:
            if isinstance(text, str):
                with profile.span("frontend.split") as split_span:
                    text = (
                        re.split(split_pattern, text.strip())
                        if split_pattern
                        else [text]
                    )
                    split_span.attrs["chunks"] = len(text)

            for graphemes_index, graphemes in enumerate(text):
                if not graphemes.strip():
                    continue

                for chunk in self.chunk_text(graphemes):
                    with profile.span("frontend.g2p") as g2p_span:
                        phonemes, _ = self.g2p(chunk)
                        g2p_span.attrs["input_chars"] = len(chunk)

                    with profile.span("frontend.tokenize"):
                        phoneme_chunks = list(self.split_phonemes_to_context(phonemes))

                    for phoneme_chunk in phoneme_chunks:
                        if self.phoneme_id_count(phoneme_chunk):
                            prepared_items.append(
                                self.prepare_phonemes(
                                    chunk,
                                    phoneme_chunk,
                                    voice,
                                    speed,
                                    text_index=graphemes_index,
                                    profile=profile,
                                )
                            )

            total_span.attrs["prepared_chunks"] = len(prepared_items)

        for prepared in prepared_items:
            yield prepared

    __call__ = prepare
