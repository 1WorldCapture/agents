from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, replace

import aiohttp
import edge_tts
from edge_tts.typing import Voice

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
MIME_TYPE = "audio/mpeg"
DEFAULT_VOICE = "en-US-EmmaMultilingualNeural"


@dataclass
class _TTSOptions:
    voice: str
    rate: str
    volume: str
    pitch: str
    proxy: str | None
    receive_timeout: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        proxy: str | None = None,
        receive_timeout: int = 60,
    ) -> None:
        """Create a Microsoft Edge online text-to-speech instance.

        Args:
            voice: Edge neural voice name. Defaults to ``en-US-EmmaMultilingualNeural``.
            rate: Speech rate adjustment, for example ``+20%`` or ``-10%``.
            volume: Volume adjustment, for example ``+10%`` or ``-10%``.
            pitch: Pitch adjustment, for example ``+10Hz`` or ``-10Hz``.
            proxy: Optional proxy URL. Defaults to ``EDGE_TTS_PROXY``.
            receive_timeout: WebSocket receive timeout in seconds for each text chunk.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            proxy=proxy or os.environ.get("EDGE_TTS_PROXY"),
            receive_timeout=receive_timeout,
        )

    @property
    def model(self) -> str:
        return "edge-tts"

    @property
    def provider(self) -> str:
        return "Microsoft Edge"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        rate: NotGivenOr[str] = NOT_GIVEN,
        volume: NotGivenOr[str] = NOT_GIVEN,
        pitch: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update options used by synthesis requests started after this call."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(rate):
            self._opts.rate = rate
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pitch):
            self._opts.pitch = pitch

    async def list_voices(self) -> list[Voice]:
        """Return the currently available Edge neural voices."""
        return await edge_tts.list_voices(proxy=self._opts.proxy)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = uuid.uuid4().hex
        communicate = edge_tts.Communicate(
            self.input_text,
            voice=self._opts.voice,
            rate=self._opts.rate,
            volume=self._opts.volume,
            pitch=self._opts.pitch,
            proxy=self._opts.proxy,
            connect_timeout=max(1, int(self._conn_options.timeout)),
            receive_timeout=self._opts.receive_timeout,
        )
        audio_received = False
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE,
        )
        try:
            async for chunk in communicate.stream():
                if chunk["type"] != "audio":
                    continue
                data = chunk.get("data")
                if data:
                    audio_received = True
                    output_emitter.push(data)
            output_emitter.flush()
        except asyncio.TimeoutError as error:
            raise APITimeoutError(
                "Edge TTS request timed out", retryable=not audio_received
            ) from error
        except aiohttp.ClientResponseError as error:
            raise APIStatusError(
                message=error.message,
                status_code=error.status,
                request_id=request_id,
                body=None,
                retryable=not audio_received,
            ) from error
        except edge_tts.exceptions.NoAudioReceived as error:
            raise APIStatusError(
                f"Edge TTS returned no audio: {error}",
                status_code=500,
                request_id=request_id,
                retryable=not audio_received,
            ) from error
        except (aiohttp.ClientError, edge_tts.exceptions.WebSocketError) as error:
            raise APIConnectionError(
                "Edge TTS websocket failed", retryable=not audio_received
            ) from error
        except Exception as error:
            raise APIConnectionError(
                "Edge TTS synthesis failed", retryable=not audio_received
            ) from error
