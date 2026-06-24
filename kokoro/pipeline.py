from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from misaki import en, espeak
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import json
import re
import torch

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

LANG_CODES = {
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


class KPipeline:
    """
    Host text frontend.

    This class performs text/G2P/chunking/vocab lookup/voice selection and
    returns exact-length numeric tensors suitable for KokoroTextDuration. It does
    not own or execute the neural inference backend.
    """

    def __init__(
        self,
        lang_code: str,
        repo_id: Optional[str] = None,
        vocab: Optional[dict[str, int]] = None,
        context_length: Optional[int] = None,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
    ):
        if repo_id is None:
            repo_id = "hexgrad/Kokoro-82M"
            print(
                f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning."
            )
        self.repo_id: str = repo_id

        if vocab is None or context_length is None:
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config_path, "r", encoding="utf-8") as r:
                config_data: dict[str, Any] = json.load(r)

            if vocab is None:
                vocab = config_data["vocab"]

            if context_length is None:
                plbert = config_data.get("plbert", {})
                if isinstance(plbert, dict):
                    context_length = plbert.get("max_position_embeddings", 512)
                else:
                    context_length = 512

        if vocab is None or context_length is None:
            raise ValueError("vocab and context_length are required")
        if context_length <= 2:
            raise ValueError(
                f"context_length must be greater than 2, got {context_length}"
            )

        lang_code = ALIASES.get(lang_code.lower(), lang_code.lower())
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)

        self.lang_code = lang_code
        self.vocab: dict[str, int] = vocab
        self.context_length: int = int(context_length)
        self.max_phoneme_len: int = self.context_length - 2
        self.voices: dict[str, torch.Tensor] = {}

        if lang_code in "ab":
            try:
                fallback = espeak.EspeakFallback(british=lang_code == "b")
            except Exception as e:
                logger.warning("EspeakFallback not enabled: OOD words will be skipped")
                logger.warning(str(e))
                fallback = None
            self.g2p = en.G2P(
                trf=trf, british=lang_code == "b", fallback=fallback, unk=""
            )
        elif lang_code == "j":
            try:
                from misaki import ja

                self.g2p = ja.JAG2P()
            except ImportError:
                logger.error(
                    "You need to `pip install misaki[ja]` to use lang_code='j'"
                )
                raise
        elif lang_code == "z":
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
            language = LANG_CODES[lang_code]
            logger.warning(
                f"Using EspeakG2P(language='{language}'). Long text is chunked by host-side sentence splitting."
            )
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        f = (
            voice
            if voice.endswith(".pt")
            else hf_hub_download(self.repo_id, filename=f"voices/{voice}.pt")
        )
        if not voice.endswith(".pt") and not voice.startswith(self.lang_code):
            logger.warning(
                f"Language mismatch, loading {voice} voice into {self.lang_code} pipeline."
            )
        pack = torch.load(f, weights_only=True)

        if len(pack.shape) == 1:
            pack = pack.unsqueeze(0)
        elif len(pack.shape) > 2:
            pack = pack.view(-1, pack.shape[-1])

        self.voices[voice] = pack
        return pack

    def load_voice(
        self, voice: Union[str, torch.Tensor], delimiter: str = ","
    ) -> torch.Tensor:
        if isinstance(voice, torch.Tensor):
            return voice
        if voice in self.voices:
            return self.voices[voice]
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            self.voices[voice] = packs[0]
        else:
            self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]

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
            yielded_count = self.phoneme_id_count(KPipeline.tokens_to_ps(tokens[:z]))
            if next_count - yielded_count <= self.max_phoneme_len:
                return z
        return len(tokens)

    def en_tokenize(
        self, tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        tks: List[en.MToken] = []

        for t in tokens:
            t.phonemes = t.phonemes or ""
            next_count = self.phoneme_id_count(KPipeline.tokens_to_ps([*tks, t]))

            if next_count > self.max_phoneme_len and tks:
                z = self.waterfall_last(tks, next_count)
                yield KPipeline.tokens_to_text(tks[:z]), KPipeline.tokens_to_ps(
                    tks[:z]
                ), tks[:z]
                tks = tks[z:]

            tks.append(t)

        if tks:
            yield KPipeline.tokens_to_text(tks), KPipeline.tokens_to_ps(tks), tks

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

    @dataclass
    class PreparedInput:
        graphemes: str
        phonemes: str
        input_ids: torch.Tensor
        input_length: int
        ref_s: torch.Tensor
        speed: float
        tokens: Optional[List[en.MToken]] = None
        text_index: Optional[int] = None

    def prepare_phonemes(
        self,
        graphemes: str,
        phonemes: str,
        voice: Union[str, torch.Tensor],
        speed: Union[float, Callable[[int], float]] = 1,
        tokens: Optional[List[en.MToken]] = None,
        text_index: Optional[int] = None,
    ) -> "KPipeline.PreparedInput":
        if not phonemes:
            raise ValueError("Cannot prepare empty phoneme string")

        ids = self.phonemes_to_ids(phonemes)
        if len(ids) > self.max_phoneme_len:
            raise ValueError(
                f"Tokenized phoneme payload too long: {len(ids)} > {self.max_phoneme_len}"
            )

        input_ids = torch.tensor([0, *ids, 0], dtype=torch.long)
        input_length = int(input_ids.numel())

        pack = self.load_voice(voice).float()
        ref_index = min(len(phonemes) - 1, pack.shape[0] - 1)
        speed_value = speed(len(phonemes)) if callable(speed) else speed

        return self.PreparedInput(
            graphemes=graphemes,
            phonemes=phonemes,
            tokens=tokens,
            text_index=text_index,
            input_ids=input_ids,
            input_length=input_length,
            ref_s=pack[ref_index],
            speed=float(speed_value),
        )

    def prepare_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: Union[str, torch.Tensor],
        speed: Union[float, Callable[[int], float]] = 1,
    ) -> Generator["KPipeline.PreparedInput", None, None]:
        if isinstance(tokens, str):
            if self.phoneme_id_count(tokens) > self.max_phoneme_len:
                raise ValueError(
                    f"Phoneme payload too long: {self.phoneme_id_count(tokens)} > {self.max_phoneme_len}"
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
    ) -> Generator["KPipeline.PreparedInput", None, None]:
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

    @staticmethod
    def join_timestamps(tokens: List[en.MToken], pred_dur: torch.Tensor):
        divisor = 80
        if not tokens or len(pred_dur) < 3:
            return

        left = right = 2 * max(0, pred_dur[0].item() - 3)
        i = 1
        for t in tokens:
            if i >= len(pred_dur) - 1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue

            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break

            t.start_ts = left / divisor
            token_dur = pred_dur[i:j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / divisor
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)
