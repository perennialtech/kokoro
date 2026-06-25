import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import torch
from loguru import logger
from misaki import en, espeak

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


def infer_language_from_voice(language: Optional[str], voice: Union[str, torch.Tensor]) -> str:
    if language:
        return normalize_language_code(language)

    if not isinstance(voice, str):
        raise ValueError("--language is required when voice is a tensor")

    stem = Path(voice).stem if voice.endswith(".pt") else voice
    if not stem:
        raise ValueError("--language is required when language cannot be inferred from voice")

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

    def load_single_voice(self, voice: str, lang_code: Optional[str] = None) -> torch.Tensor:
        if voice in self.cpu_cache:
            return self.cpu_cache[voice]

        path = self._voice_path(voice)
        name = path.stem

        if lang_code is not None and not name.startswith(lang_code):
            logger.warning(
                f"Language mismatch, loading {name} voice into {lang_code} pipeline."
            )

        pack = self._normalize_pack(torch.load(path, weights_only=True))
        self.cpu_cache[voice] = pack
        return pack

    def _load_cpu(
        self,
        voice: Union[str, torch.Tensor],
        *,
        lang_code: Optional[str],
        delimiter: str = ",",
    ) -> torch.Tensor:
        if isinstance(voice, torch.Tensor):
            return self._normalize_pack(voice)

        if voice in self.cpu_cache:
            return self.cpu_cache[voice]

        packs = [
            self.load_single_voice(v.strip(), lang_code=lang_code)
            for v in voice.split(delimiter)
            if v.strip()
        ]
        if not packs:
            raise ValueError("voice must not be empty")

        self.cpu_cache[voice] = packs[0] if len(packs) == 1 else torch.mean(torch.stack(packs), dim=0)
        return self.cpu_cache[voice]

    def load(
        self,
        voice: Union[str, torch.Tensor],
        *,
        lang_code: Optional[str] = None,
        delimiter: str = ",",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        base = self._load_cpu(voice, lang_code=lang_code, delimiter=delimiter)

        target_device = torch.device(device) if device is not None else self.target_device
        target_dtype = dtype or self.target_dtype

        if target_device is None and target_dtype is None:
            return base

        if target_device is None:
            target_device = base.device
        if target_dtype is None:
            target_dtype = base.dtype

        if isinstance(voice, torch.Tensor):
            return base.to(device=target_device, dtype=target_dtype)

        key = (voice, str(target_device), str(target_dtype))
        cached = self.device_cache.get(key)
        if cached is not None:
            return cached

        cached = base.to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=target_device.type == "cuda",
        )
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
            raise ValueError(f"context_length must be greater than 2, got {context_length}")

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
                logger.error("You need to `pip install misaki[ja]` to use lang_code='j'")
                raise
        elif self.lang_code == "z":
            try:
                from misaki import zh

                self.g2p = zh.ZHG2P(
                    version=None if repo_id.endswith("/Kokoro-82M") else "1.1",
                    en_callable=en_callable,
                )
            except ImportError:
                logger.error("You need to `pip install misaki[zh]` to use lang_code='z'")
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
    ) -> PreparedInput:
        if not phonemes:
            raise ValueError("Cannot prepare empty phoneme string")

        ids = self.phonemes_to_ids(phonemes)
        if len(ids) > self.max_phoneme_len:
            raise ValueError(
                f"Tokenized phoneme payload too long: {len(ids)} > {self.max_phoneme_len}"
            )

        pack = self.voice_store.load(voice, lang_code=self.lang_code)
        ref_index = min(len(phonemes) - 1, pack.shape[0] - 1)
        speed_value = speed(len(phonemes)) if callable(speed) else speed

        return PreparedInput(
            graphemes=graphemes,
            phonemes=phonemes,
            tokens=tokens,
            text_index=text_index,
            input_ids=torch.tensor([[0, *ids, 0]], dtype=torch.long),
            ref_s=pack[ref_index].reshape(1, -1).contiguous(),
            speed=torch.tensor([float(speed_value)], dtype=torch.float32, device=pack.device),
        )

    def prepare_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: Union[str, torch.Tensor],
        speed: Union[float, Callable[[int], float]] = 1,
    ) -> Generator[PreparedInput, None, None]:
        if isinstance(tokens, str):
            if self.phoneme_id_count(tokens) > self.max_phoneme_len:
                raise ValueError(
                    f"Phoneme payload too long: {self.phoneme_id_count(tokens)} > "
                    f"{self.max_phoneme_len}"
                )
            yield self.prepare_phonemes("", tokens, voice, speed)
            return

        for gs, ps, tks in self.en_tokenize(tokens):
            if ps:
                yield self.prepare_phonemes(gs, ps, voice, speed, tokens=tks)

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
    ) -> Generator[PreparedInput, None, None]:
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():
                continue

            if self.lang_code in "ab":
                _, tokens = self.g2p(graphemes)
                if tokens is None:
                    continue
                for gs, ps, tks in self.en_tokenize(tokens):
                    if ps:
                        yield self.prepare_phonemes(
                            gs,
                            ps,
                            voice,
                            speed,
                            tokens=tks,
                            text_index=graphemes_index,
                        )
                continue

            for chunk in self.chunk_non_english_text(graphemes):
                ps, _ = self.g2p(chunk)
                for ps_chunk in self.split_phonemes_to_context(ps):
                    if ps_chunk:
                        yield self.prepare_phonemes(
                            chunk,
                            ps_chunk,
                            voice,
                            speed,
                            text_index=graphemes_index,
                        )

    __call__ = prepare
