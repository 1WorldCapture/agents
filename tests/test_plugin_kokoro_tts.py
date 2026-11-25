from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.kokoro import TTS, tts as kokoro_tts

pytestmark = pytest.mark.unit


class _FakeKokoro:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create_stream(
        self, phonemes: str, **kwargs: Any
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        self.calls.append({"phonemes": phonemes, **kwargs})
        for chunk in self.chunks:
            yield chunk, kokoro_tts.SAMPLE_RATE


class _FakeEngine:
    def __init__(self, phonemes: str, chunks: list[np.ndarray]) -> None:
        self.kokoro = _FakeKokoro(chunks)
        self.phonemes = phonemes
        self.texts: list[str] = []

    def phonemize(self, text: str) -> str:
        self.texts.append(text)
        return self.phonemes


def _use(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    def load(directory: Path) -> Any:
        if isinstance(engine, Exception):
            raise engine
        return engine

    monkeypatch.setattr(kokoro_tts, "_engine", load)


async def _frames(provider: TTS, text: str) -> list[Any]:
    async with provider.synthesize(text, conn_options=APIConnectOptions(max_retry=2)) as stream:
        return [event.frame async for event in stream]


async def test_streams_every_chunk_as_pcm_with_the_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 0.3 s then 0.2 s; samples past ±1 must clip, not wrap around
    chunks = [np.full(7200, 2.0, dtype=np.float32), np.full(4800, -0.5, dtype=np.float32)]
    engine = _FakeEngine("ㄋㄧ2ㄏㄠ3", chunks)
    _use(monkeypatch, engine)
    provider = TTS(voice="zm_009", speed=1.2, model_dir=tmp_path)
    provider.update_options(voice="zf_002")

    frames = await _frames(provider, "你好")

    assert engine.texts == ["你好"]
    assert engine.kokoro.calls == [
        {"phonemes": "ㄋㄧ2ㄏㄠ3", "voice": "zf_002", "speed": 1.2, "is_phonemes": True}
    ]
    samples = np.concatenate([np.frombuffer(frame.data, dtype=np.int16) for frame in frames])
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in frames)
    # AudioEmitter ends each segment with a little synthetic silence
    assert len(samples) >= 12000 and not samples[12000:].any()
    assert (samples[:7200] == 32767).all() and (samples[7200:12000] == -16383).all()


async def test_text_without_speech_gives_short_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _FakeEngine("... ?", [])
    _use(monkeypatch, engine)

    frames = await _frames(TTS(model_dir=tmp_path), "……？")

    samples = np.concatenate([np.frombuffer(frame.data, dtype=np.int16) for frame in frames])
    assert engine.kokoro.calls == []
    assert 1200 <= len(samples) <= 2400 and not samples.any()


async def test_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    attempts = []

    def load(directory: Path) -> Any:
        attempts.append(directory)
        raise FileNotFoundError("kokoro-v1.1-zh.onnx")

    monkeypatch.setattr(kokoro_tts, "_engine", load)

    with pytest.raises(APIConnectionError, match="kokoro-v1.1-zh.onnx"):
        await _frames(TTS(model_dir=tmp_path), "你好")
    assert attempts == [tmp_path]


async def test_long_sentence_plays_clause_by_clause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 0.1 s of speech per clause; the fake G2P returns the text as its phonemes
    engine = _FakeEngine("", [np.full(2400, 0.5, dtype=np.float32)])
    engine.phonemize = lambda text: engine.texts.append(text) or text  # type: ignore[method-assign]
    _use(monkeypatch, engine)

    text = "这是一段比较长的回答，好，价格是1,000元，时间10:30; Python and LiveKit, ok。"
    frames = await _frames(TTS(model_dir=tmp_path), text)

    # a clause too short to sound right on its own joins the next one, the short last one
    # joins the one before, and ASCII marks without a space after them do not split
    assert engine.texts == [
        "这是一段比较长的回答，",
        "好，价格是1,000元，",
        "时间10:30;",
        "Python and LiveKit, ok。",
    ]
    assert [call["phonemes"] for call in engine.kokoro.calls] == engine.texts
    samples = np.concatenate([np.frombuffer(frame.data, dtype=np.int16) for frame in frames])
    speech = samples != 0
    assert speech[:2400].all() and not speech[2400:4800].any() and speech[4800:7200].all()
