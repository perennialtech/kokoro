import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import torch
from loguru import logger
from misaki import en, espeak

from .telemetry import (NoOpProfileContext, ProfileContext,
                        normalize_voice_label)

ALIASES = {
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

LANGUAGE_CODES = {
    "a": "American English",
    "b": "British English",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "j": "Japanese",
    "z": "Mandarin Chinese",
}


def normalize_language_code(lang_code: str) -> str:
    lang_code = ALIASES.get(lang_code.lower(), lang_code.lower())
    if lang_code not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language code {lang_code!r}")
    return lang_code


def infer_language_from_voice(
    language: Optional[str], voice: Union[str, torch.Tensor]
) -> str:
    if language:
        return normalize_language_code(language)

    if not isinstance(voice, str):
        raise ValueError("--language is required when voice is a tensor")

    stem = Path(voice).stem if voice.endswith(".pt") else voice
    if not stem:
        raise ValueError(
            "--language is required when language cannot be inferred from voice"
        )

    return normalize_language_code(stem[0])


@dataclass
class PreparedInput:
    graphemes: str
    phonemes: str
    input_ids: torch.Tensor
    ref_s: torch.Tensor
    speed: torch.Tensor
    tokens: Optional[List[en.MToken]] = None
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

    def _voice_path(self, voice: str) -> Path:
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
        lang_code: Optional[str] = None,
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
            name = path.stem

        if lang_code is not None and not name.startswith(lang_code):
            logger.warning(
                f"Language mismatch, loading {name} voice into {lang_code} pipeline."
            )

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
        voice: Union[str, torch.Tensor],
        *,
        lang_code: Optional[str],
        delimiter: str = ",",
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        voice_label, voice_kind = normalize_voice_label(voice)

        if isinstance(voice, torch.Tensor):
            return self._normalize_pack(voice)

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
            self.load_single_voice(v.strip(), lang_code=lang_code, profile=profile)
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
        voice: Union[str, torch.Tensor],
        *,
        lang_code: Optional[str] = None,
        delimiter: str = ",",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        profile: Optional[ProfileContext] = None,
    ) -> torch.Tensor:
        profile = profile or NoOpProfileContext()
        voice_label, voice_kind = normalize_voice_label(voice)
        base = self._load_cpu(
            voice,
            lang_code=lang_code,
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

        if isinstance(voice, torch.Tensor):
            with profile.span(
                "voice.transfer_device",
                cuda=target_device.type == "cuda",
                attrs={"voice_kind": "tensor"},
            ):
                return base.to(device=target_device, dtype=target_dtype)

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
        lang_code: str,
        repo_id: str,
        vocab: dict[str, int],
        context_length: int,
        voice_store: VoiceStore,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
    ):
        if context_length <= 2:
            raise ValueError(
                f"context_length must be greater than 2, got {context_length}"
            )

        self.repo_id = repo_id
        self.lang_code = normalize_language_code(lang_code)
        self.vocab = vocab
        self.context_length = int(context_length)
        self.max_phoneme_len = self.context_length - 2
        self.voice_store = voice_store

        if self.lang_code in "ab":
            try:
                fallback = espeak.EspeakFallback(british=self.lang_code == "b")
            except Exception as e:
                logger.warning("EspeakFallback not enabled: OOD words will be skipped")
                logger.warning(str(e))
                fallback = None
            self.g2p = en.G2P(
                trf=trf,
                british=self.lang_code == "b",
                fallback=fallback,
                unk="",
            )
        elif self.lang_code == "j":
            try:
                from misaki import ja

                self.g2p = ja.JAG2P()
            except ImportError:
                logger.error(
                    "You need to `pip install misaki[ja]` to use lang_code='j'"
                )
                raise
        elif self.lang_code == "z":
            try:
                from misaki import zh

                self.g2p = zh.ZHG2P(
                    version=None if repo_id.endswith("/Kokoro-82M") else "1.1",
                    en_callable=en_callable,
                )
            except ImportError:
                logger.error(
                    "You need to `pip install misaki[zh]` to use lang_code='z'"
                )
                raise
        else:
            language = LANGUAGE_CODES[self.lang_code]
            logger.warning(
                f"Using EspeakG2P(language='{language}'). Long text is chunked by "
                "host-side sentence splitting."
            )
            self.g2p = espeak.EspeakG2P(language=language)

    @staticmethod
    def tokens_to_ps(tokens: List[en.MToken]) -> str:
        return "".join(
            (t.phonemes or "") + (" " if t.whitespace else "") for t in tokens
        ).strip()

    @staticmethod
    def tokens_to_text(tokens: List[en.MToken]) -> str:
        return "".join(t.text + t.whitespace for t in tokens).strip()

    def phonemes_to_ids(self, phonemes: str) -> List[int]:
        return [self.vocab[p] for p in phonemes if self.vocab.get(p) is not None]

    def phoneme_id_count(self, phonemes: str) -> int:
        return sum(1 for p in phonemes if self.vocab.get(p) is not None)

    def waterfall_last(
        self,
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ["!.?…", ":;", ",—"],
        bumps: List[str] = [")", "”"],
    ) -> int:
        for w in waterfall:
            z = next(
                (
                    i
                    for i, t in reversed(list(enumerate(tokens)))
                    if t.phonemes in set(w)
                ),
                None,
            )
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            yielded_count = self.phoneme_id_count(TextFrontend.tokens_to_ps(tokens[:z]))
            if next_count - yielded_count <= self.max_phoneme_len:
                return z
        return len(tokens)

    def en_tokenize(
        self,
        tokens: List[en.MToken],
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        tks: List[en.MToken] = []

        for t in tokens:
            t.phonemes = t.phonemes or ""
            next_count = self.phoneme_id_count(TextFrontend.tokens_to_ps([*tks, t]))

            if next_count > self.max_phoneme_len and tks:
                z = self.waterfall_last(tks, next_count)
                yield (
                    TextFrontend.tokens_to_text(tks[:z]),
                    TextFrontend.tokens_to_ps(tks[:z]),
                    tks[:z],
                )
                tks = tks[z:]

            tks.append(t)

        if tks:
            yield TextFrontend.tokens_to_text(tks), TextFrontend.tokens_to_ps(tks), tks

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
        voice: Union[str, torch.Tensor],
        speed: Union[float, Callable[[int], float]] = 1,
        tokens: Optional[List[en.MToken]] = None,
        text_index: Optional[int] = None,
        profile: Optional[ProfileContext] = None,
    ) -> PreparedInput:
        profile = profile or NoOpProfileContext()
        with profile.span("frontend.prepare_phonemes") as span:
            if not phonemes:
                raise ValueError("Cannot prepare empty phoneme string")

            ids = self.phonemes_to_ids(phonemes)
            if len(ids) > self.max_phoneme_len:
                raise ValueError(
                    f"Tokenized phoneme payload too long: {len(ids)} > {self.max_phoneme_len}"
                )

            pack = self.voice_store.load(
                voice,
                lang_code=self.lang_code,
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
                tokens=tokens,
                text_index=text_index,
                input_ids=torch.tensor([[0, *ids, 0]], dtype=torch.long),
                ref_s=pack[ref_index].reshape(1, -1).contiguous(),
                speed=torch.tensor(
                    [float(speed_value)], dtype=torch.float32, device=pack.device
                ),
            )

    def prepare_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: Union[str, torch.Tensor],
        speed: Union[float, Callable[[int], float]] = 1,
        profile: Optional[ProfileContext] = None,
    ) -> Generator[PreparedInput, None, None]:
        profile = profile or NoOpProfileContext()

        if isinstance(tokens, str):
            if self.phoneme_id_count(tokens) > self.max_phoneme_len:
                raise ValueError(
                    f"Phoneme payload too long: {self.phoneme_id_count(tokens)} > "
                    f"{self.max_phoneme_len}"
                )
            yield self.prepare_phonemes("", tokens, voice, speed, profile=profile)
            return

        with profile.span("frontend.tokenize"):
            chunks = list(self.en_tokenize(tokens))

        for gs, ps, tks in chunks:
            if ps:
                yield self.prepare_phonemes(
                    gs,
                    ps,
                    voice,
                    speed,
                    tokens=tks,
                    profile=profile,
                )

    @staticmethod
    def chunk_non_english_text(text: str, chunk_size: int = 400) -> List[str]:
        chunks = []
        sentences = re.split(r"([.!?]+)", text)
        current = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if len(current) + len(sentence) <= chunk_size:
                current += sentence
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks or [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]

    def prepare(
        self,
        text: Union[str, List[str]],
        voice: Union[str, torch.Tensor],
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

                if self.lang_code in "ab":
                    with profile.span("frontend.g2p") as g2p_span:
                        _, tokens = self.g2p(graphemes)
                        g2p_span.attrs["input_chars"] = len(graphemes)
                    if tokens is None:
                        continue

                    with profile.span("frontend.tokenize"):
                        tokenized = list(self.en_tokenize(tokens))

                    for gs, ps, tks in tokenized:
                        if ps:
                            prepared_items.append(
                                self.prepare_phonemes(
                                    gs,
                                    ps,
                                    voice,
                                    speed,
                                    tokens=tks,
                                    text_index=graphemes_index,
                                    profile=profile,
                                )
                            )
                    continue

                for chunk in self.chunk_non_english_text(graphemes):
                    with profile.span("frontend.g2p") as g2p_span:
                        ps, _ = self.g2p(chunk)
                        g2p_span.attrs["input_chars"] = len(chunk)

                    with profile.span("frontend.tokenize"):
                        ps_chunks = list(self.split_phonemes_to_context(ps))

                    for ps_chunk in ps_chunks:
                        if ps_chunk:
                            prepared_items.append(
                                self.prepare_phonemes(
                                    chunk,
                                    ps_chunk,
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
