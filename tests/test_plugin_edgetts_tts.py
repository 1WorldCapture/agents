from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import aiohttp
import edge_tts
import pytest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)
from livekit.plugins.edgetts import TTS, tts as edgetts_tts

pytestmark = pytest.mark.unit

_MP3_DATA = (Path(__file__).parent / "long.mp3").read_bytes()


class _FakeCommunicate:
    script: list[dict[str, Any] | Exception] = []
    instances: list[_FakeCommunicate] = []

    def __init__(self, text: str, **kwargs: Any) -> None:
        self.text = text
        self.kwargs = kwargs
        type(self).instances.append(self)

    def stream(self) -> AsyncIterator[dict[str, Any]]:
        async def generate() -> AsyncIterator[dict[str, Any]]:
            for item in type(self).script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return generate()


@pytest.fixture
def fake_communicate(monkeypatch: pytest.MonkeyPatch) -> type[_FakeCommunicate]:
    _FakeCommunicate.script = []
    _FakeCommunicate.instances = []
    monkeypatch.setattr(edgetts_tts.edge_tts, "Communicate", _FakeCommunicate)
    return _FakeCommunicate


async def _collect(
    provider: TTS,
    text: str = "hello",
    conn_options: APIConnectOptions | None = None,
) -> list[Any]:
    async with provider.synthesize(
        text, conn_options=conn_options or APIConnectOptions(max_retry=0)
    ) as stream:
        return [event async for event in stream]


async def test_synthesis_options_metadata_and_mp3_decoding(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    split = len(_MP3_DATA) // 2
    fake_communicate.script = [
        {
            "type": "SentenceBoundary",
            "offset": 0,
            "duration": 1_000_000,
            "text": "hello",
        },
        {"type": "audio", "data": _MP3_DATA[:split]},
        {"type": "audio", "data": _MP3_DATA[split:]},
    ]
    provider = TTS(
        voice="zh-CN-XiaoxiaoNeural",
        rate="+20%",
        volume="-10%",
        pitch="+5Hz",
        proxy="http://proxy.test:8080",
        receive_timeout=12,
    )
    events = await _collect(
        provider,
        text="你好",
        conn_options=APIConnectOptions(max_retry=0, timeout=7),
    )

    assert len(fake_communicate.instances) == 1
    request = fake_communicate.instances[0]
    assert request.text == "你好"
    assert request.kwargs == {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+20%",
        "volume": "-10%",
        "pitch": "+5Hz",
        "proxy": "http://proxy.test:8080",
        "connect_timeout": 7,
        "receive_timeout": 12,
    }
    assert events
    assert all(event.frame.sample_rate == 24000 for event in events)
    assert all(event.frame.num_channels == 1 for event in events)
    assert sum(event.frame.duration for event in events) > 0
    assert events[-1].is_final
    assert provider.model == "edge-tts"
    assert provider.provider == "Microsoft Edge"
    assert not provider.capabilities.streaming


async def test_update_options_applies_to_next_request(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    fake_communicate.script = [{"type": "audio", "data": _MP3_DATA}]
    provider = TTS()
    provider.update_options(voice="en-US-AriaNeural", rate="-5%", volume="+5%", pitch="-2Hz")
    await _collect(provider)
    assert fake_communicate.instances[0].kwargs["voice"] == "en-US-AriaNeural"
    assert fake_communicate.instances[0].kwargs["rate"] == "-5%"
    assert fake_communicate.instances[0].kwargs["volume"] == "+5%"
    assert fake_communicate.instances[0].kwargs["pitch"] == "-2Hz"


async def test_no_audio_is_retryable_before_audio_output(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    fake_communicate.script = [edge_tts.exceptions.NoAudioReceived("no audio")]
    provider = TTS()
    with pytest.raises(APIStatusError, match="no audio") as caught:
        await _collect(
            provider,
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0),
        )
    assert caught.value.status_code == 500
    assert len(fake_communicate.instances) == 2


async def test_http_403_is_mapped_to_api_status_error(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    fake_communicate.script = [
        aiohttp.ClientResponseError(
            request_info=Mock(), history=(), status=403, message="Forbidden"
        )
    ]
    with pytest.raises(APIStatusError, match="Forbidden") as caught:
        await _collect(TTS())
    assert caught.value.status_code == 403
    assert not caught.value.retryable


async def test_timeout_is_mapped_to_api_timeout(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    fake_communicate.script = [TimeoutError("timed out")]
    with pytest.raises(APITimeoutError, match="timed out"):
        await _collect(TTS())


async def test_connection_failure_after_audio_is_not_retried(
    fake_communicate: type[_FakeCommunicate],
) -> None:
    fake_communicate.script = [
        {"type": "audio", "data": _MP3_DATA},
        aiohttp.ClientConnectionError("connection lost"),
    ]
    with pytest.raises(APIConnectionError, match="websocket failed") as caught:
        await _collect(TTS(), conn_options=APIConnectOptions(max_retry=3, retry_interval=0))
    assert not caught.value.retryable
    assert len(fake_communicate.instances) == 1


async def test_list_voices_uses_configured_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = {
        "Name": "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)",
        "ShortName": "zh-CN-XiaoxiaoNeural",
        "Gender": "Female",
        "Locale": "zh-CN",
        "SuggestedCodec": "audio-24khz-48kbitrate-mono-mp3",
        "FriendlyName": "Microsoft Xiaoxiao Online (Natural) - Chinese (Mainland)",
        "Status": "GA",
        "VoiceTag": {"ContentCategories": [], "VoicePersonalities": []},
    }

    async def fake_list_voices(*, proxy: str | None = None) -> list[Any]:
        assert proxy == "http://proxy.test:8080"
        return [voice]

    monkeypatch.setattr(edgetts_tts.edge_tts, "list_voices", fake_list_voices)
    voices = await TTS(proxy="http://proxy.test:8080").list_voices()
    assert voices[0]["ShortName"] == "zh-CN-XiaoxiaoNeural"
