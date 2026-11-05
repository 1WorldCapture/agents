from __future__ import annotations

import asyncio
import os
import time
import uuid
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import aiohttp
import av
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._client import DEFAULT_WS_URL, Identity, IdentityStore, VoiceResult, WeTypeClient

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLES_PER_FRAME = SAMPLE_RATE // 50
PACKETS_PER_REQUEST = 6
FINAL_POLL_ATTEMPTS = 8
FINAL_POLL_INTERVAL = 0.25
_OPUS_HEADER = b"#!OPUS_RAW_V1" + bytes((2, 1, 0))


@dataclass
class _STTOptions:
    language: str
    interim_results: bool


class _OpusEncoder:
    def __init__(self) -> None:
        try:
            encoder = av.CodecContext.create("libopus", "w")
            encoder.options["application"] = "audio"
            encoder.options["complexity"] = "9"
        except av.codec.codec.UnknownCodecError:
            encoder = av.CodecContext.create("opus", "w")
            encoder.options["strict"] = "-2"
        encoder.sample_rate = SAMPLE_RATE
        encoder.format = "s16"
        encoder.layout = "mono"
        encoder.bit_rate = 64000
        encoder.open()
        self._encoder = encoder

    def encode(self, frame: rtc.AudioFrame) -> list[bytes]:
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(1, -1)
        av_frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        av_frame.sample_rate = SAMPLE_RATE
        return [bytes(packet) for packet in self._encoder.encode(av_frame)]


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "zh-CN",
        interim_results: bool = True,
        device: str | None = None,
        uin: str | None = None,
        identity_cache_path: str | Path | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an online Weixin speech recognizer.

        Args:
            language: Language attached to transcript events. Defaults to ``zh-CN``.
            interim_results: Emit replace-style interim transcripts.
            device: Existing Weixin input-method device identity. Defaults to
                ``WEIXIN_DEVICE``. Must be provided together with ``uin``.
            uin: Existing Weixin input-method UIN. Defaults to ``WEIXIN_UIN``.
            identity_cache_path: Identity cache location. Defaults to
                ``~/.livekit/weixin-identity.json`` or ``WEIXIN_IDENTITY_CACHE_PATH``.
            base_url: WebSocket endpoint. Defaults to ``WEIXIN_WS_URL`` or the Weixin
                online input-method endpoint.
            http_session: Optional aiohttp session to reuse.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                offline_recognize=True,
            )
        )
        self._opts = _STTOptions(language=language, interim_results=interim_results)
        device = device or os.environ.get("WEIXIN_DEVICE")
        uin = uin or os.environ.get("WEIXIN_UIN")
        if bool(device) != bool(uin):
            raise ValueError("Weixin device and uin must be provided together")
        identity = Identity(device=device, uin=uin) if device and uin else None
        cache_path = identity_cache_path or os.environ.get("WEIXIN_IDENTITY_CACHE_PATH")
        self._identity_store = IdentityStore(identity=identity, cache_path=cache_path)
        self._base_url = base_url or os.environ.get("WEIXIN_WS_URL") or DEFAULT_WS_URL
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "weixin-ime-online"

    @property
    def provider(self) -> str:
        return "Weixin"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        stream = self.stream(language=language, conn_options=conn_options)
        stream.push_frame(rtc.combine_audio_frames(buffer))
        stream.end_input()
        text = ""
        detected_language = LanguageCode(self._opts.language)
        try:
            async for event in stream:
                if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                    text = event.alternatives[0].text
                    detected_language = event.alternatives[0].language
        finally:
            await stream.aclose()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=detected_language, text=text)],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            session=self._ensure_session(),
            identity_store=self._identity_store,
            base_url=self._base_url,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update options used by streams opened after this call."""
        if is_given(language):
            self._opts.language = language
        if is_given(interim_results):
            self._opts.interim_results = interim_results
            self._capabilities.interim_results = interim_results

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in tuple(self._streams)))


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
        identity_store: IdentityStore,
        base_url: str,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._opts = opts
        self._session = session
        self._identity_store = identity_store
        self._base_url = base_url
        self._speaking = False
        self._input_consumed = False
        self._request_id = ""
        self._usage_duration = 0.0
        self._usage_last_flush = time.monotonic()

    async def _run(self) -> None:
        self._input_consumed = False
        self._speaking = False
        self._usage_duration = 0.0
        self._usage_last_flush = time.monotonic()
        self._request_id = uuid.uuid4().hex
        try:
            client = await WeTypeClient.open(
                session=self._session,
                base_url=self._base_url,
                store=self._identity_store,
                timeout=self._conn_options.timeout,
            )
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as error:
            raise APIConnectionError("Weixin STT handshake failed") from error

        latest_text = ""
        polished_text = ""
        packets: list[bytes] = []
        sequence = 0
        total_bytes = 0

        def handle_result(result: VoiceResult) -> None:
            nonlocal latest_text, polished_text
            if result.text and result.text != latest_text:
                latest_text = result.text
                self._emit_interim(result.text)
            if result.polished:
                polished_text = result.polished

        async def upload(batch: list[bytes], *, is_end: bool) -> None:
            nonlocal sequence, total_bytes
            sequence += 1
            framed = bytearray(_OPUS_HEADER if sequence == 1 else b"")
            for packet in batch:
                if len(packet) > 0xFFFF:
                    raise ValueError("Opus packet exceeds the Weixin framing limit")
                framed += len(packet).to_bytes(2, "little") + packet
            total_bytes += len(framed)
            result = await client.voice(
                voice_id=self._request_id,
                opus=bytes(framed),
                seq=sequence,
                total_bytes=total_bytes,
                is_end=is_end,
            )
            handle_result(result)

        try:
            encoder = _OpusEncoder()
            chunker = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                self._input_consumed = True
                mono = _to_mono(data)
                self._report_usage(mono.duration)
                for frame in chunker.push(mono.data.tobytes()):
                    packets.extend(encoder.encode(frame))
                    if len(packets) >= PACKETS_PER_REQUEST:
                        await upload(packets, is_end=False)
                        packets = []
            for frame in chunker.flush():
                packets.extend(encoder.encode(_pad_frame(frame, SAMPLES_PER_FRAME)))
            while len(packets) > PACKETS_PER_REQUEST:
                await upload(packets[:PACKETS_PER_REQUEST], is_end=False)
                packets = packets[PACKETS_PER_REQUEST:]
            await upload(packets, is_end=True)

            for _ in range(FINAL_POLL_ATTEMPTS):
                if polished_text:
                    break
                await asyncio.sleep(FINAL_POLL_INTERVAL)
                result = await client.voice(
                    voice_id=self._request_id,
                    opus=None,
                    seq=0,
                    total_bytes=total_bytes,
                    is_end=True,
                )
                handle_result(result)
            self._emit_final(polished_text or latest_text)
        except (APIConnectionError, APIStatusError, APITimeoutError) as error:
            if self._input_consumed:
                error.retryable = False
            raise
        except Exception as error:
            raise APIConnectionError(
                "Weixin STT stream failed", retryable=not self._input_consumed
            ) from error
        finally:
            self._flush_usage()
            await client.aclose()

    def _emit_interim(self, text: str) -> None:
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        if self._opts.interim_results:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        stt.SpeechData(language=LanguageCode(self._opts.language), text=text)
                    ],
                )
            )

    def _emit_final(self, text: str) -> None:
        if not text:
            return
        if not self._speaking:
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(language=LanguageCode(self._opts.language), text=text)
                ],
            )
        )
        self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        self._speaking = False

    def _report_usage(self, duration: float) -> None:
        self._usage_duration += duration
        if time.monotonic() - self._usage_last_flush >= 5.0:
            self._flush_usage()

    def _flush_usage(self) -> None:
        if self._usage_duration <= 0:
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=self._usage_duration),
            )
        )
        self._usage_duration = 0.0
        self._usage_last_flush = time.monotonic()


def _to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    if frame.num_channels == 1:
        return frame
    samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=frame.samples_per_channel,
    )


def _pad_frame(frame: rtc.AudioFrame, samples_per_channel: int) -> rtc.AudioFrame:
    if frame.samples_per_channel >= samples_per_channel:
        return frame
    return rtc.AudioFrame(
        data=frame.data.tobytes() + b"\x00\x00" * (samples_per_channel - frame.samples_per_channel),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )
