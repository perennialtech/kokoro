from unittest.mock import Mock

import pytest

from kokoro.pipeline import TextFrontend


@pytest.fixture
def frontend():
    frontend = TextFrontend.__new__(TextFrontend)
    frontend.vocab = {"a": 1, "b": 2}
    frontend.max_phoneme_len = 500
    frontend.g2p = Mock(side_effect=lambda chunk: (chunk.lower(), None))
    frontend.prepare_phonemes = Mock(
        side_effect=lambda graphemes, phonemes, voice, speed, **kwargs: (
            graphemes,
            phonemes,
            kwargs["text_index"],
        )
    )
    return frontend


def test_preparation_yields_before_processing_later_text(frontend):
    chunks = frontend.prepare("a" * 400 + "b" * 10, voice="af_heart")

    assert next(chunks) == ("a" * 400, "a" * 400, 0)
    assert frontend.g2p.call_count == 1
    assert next(chunks) == ("b" * 10, "b" * 10, 0)
    assert frontend.g2p.call_count == 2
    with pytest.raises(StopIteration):
        next(chunks)


def test_preparation_preserves_text_indices_and_skips_empty_phonemes(frontend):
    frontend.g2p.side_effect = lambda chunk: (
        "" if chunk == "missing" else chunk.lower(),
        None,
    )
    chunks = frontend.prepare(["a", "missing", " ", "b"], voice="af_heart")

    assert next(chunks) == ("a", "a", 0)
    assert frontend.g2p.call_count == 1
    assert list(chunks) == [("b", "b", 3)]
    assert frontend.g2p.call_count == 3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),
        ("abc", ["abc"]),
        ("A. B!", ["A. B!"]),
        ("a" * 400 + "b", ["a" * 400, "b"]),
        ("a" * 400 + "   ", ["a" * 400]),
    ],
)
def test_chunk_boundaries(text, expected):
    assert list(TextFrontend.chunk_text(text)) == expected
