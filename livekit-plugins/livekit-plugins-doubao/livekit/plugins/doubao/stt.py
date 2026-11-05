from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlencode

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

from ._device import DEFAULT_APP_KEY, USER_AGENT, DeviceRegistry
from ._proto import (
    FIRST_FRAME,
    LAST_FRAME,
    MIDDLE_FRAME,
    OK_STATUS,
    Response,
    Transcript,
    TranscriptUpdate,
    encode_request,
)

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLES_PER_FRAME = SAMPLE_RATE // 50
DEFAULT_WS_URL = "wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws"


@dataclass
class _STTOptions:
    language: str
    interim_results: bool
    punctuation: bool
    speech_rejection: bool


class _OpusEncoder:
    def __init__(self) -> None:
        try:
            encoder = av.CodecContext.create("libopus", "w")
            encoder.options["application"] = "voip"
            encoder.options["complexity"] = "5"
        except av.codec.codec.UnknownCodecError:
            encoder = av.CodecContext.create("opus", "w")
            encoder.options["strict"] = "-2"
        encoder.sample_rate = SAMPLE_RATE
        encoder.format = "s16"
        encoder.layout = "mono"
        encoder.bit_rate = 16000
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
        punctuation: bool = True,
        speech_rejection: bool = True,
        app_key: str | None = None,
        device_id: str | None = None,
        device_cache_path: str | Path | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an online Doubao speech recognizer.

        Args:
            language: Language attached to transcript events. Defaults to ``zh-CN``.
            interim_results: Emit replace-style interim transcripts.
            punctuation: Enable server-side punctuation.
            speech_rejection: Enable server-side non-speech rejection.
            app_key: Doubao ASR app key. Defaults to ``DOUBAO_APP_KEY`` or the
                input-method client key used by the reference implementation.
            device_id: Existing Doubao device ID. Defaults to ``DOUBAO_DEVICE_ID``.
                When omitted, a device is registered online and cached locally.
            device_cache_path: Device cache location. Defaults to
                ``~/.livekit/doubao-device.json`` or ``DOUBAO_DEVICE_CACHE_PATH``.
            base_url: WebSocket endpoint. Defaults to ``DOUBAO_WS_URL`` or the
                Doubao online input-method endpoint.
            http_session: Optional aiohttp session to reuse.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                offline_recognize=True,
            )
        )
        self._opts = _STTOptions(
            language=language,
            interim_results=interim_results,
            punctuation=punctuation,
            speech_rejection=speech_rejection,
        )
        self._app_key = app_key or os.environ.get("DOUBAO_APP_KEY") or DEFAULT_APP_KEY
        self._device_id = device_id or os.environ.get("DOUBAO_DEVICE_ID")
        self._device_cache_path = device_cache_path or os.environ.get("DOUBAO_DEVICE_CACHE_PATH")
        self._base_url = base_url or os.environ.get("DOUBAO_WS_URL") or DEFAULT_WS_URL
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()
        self._registry: DeviceRegistry | None = None

    @property
    def model(self) -> str:
        return "doubao-ime-online"

    @property
    def provider(self) -> str:
        return "Doubao"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _ensure_registry(self) -> DeviceRegistry:
        if not self._registry:
            self._registry = DeviceRegistry(
                session=self._ensure_session(),
                device_id=self._device_id,
                cache_path=self._device_cache_path,
            )
        return self._registry

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
        texts: list[str] = []
        detected_language = LanguageCode(self._opts.language)
        try:
            async for event in stream:
                if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                    texts.append(event.alternatives[0].text)
                    detected_language = event.alternatives[0].language
        finally:
            await stream.aclose()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=detected_language, text=_join_texts(texts))],
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
            registry=self._ensure_registry(),
            app_key=self._app_key,
            base_url=self._base_url,
            reset_device_on_failure=self._device_id is None,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        punctuation: NotGivenOr[bool] = NOT_GIVEN,
        speech_rejection: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update options used by streams opened after this call."""
        if is_given(language):
            self._opts.language = language
        if is_given(interim_results):
            self._opts.interim_results = interim_results
            self._capabilities.interim_results = interim_results
        if is_given(punctuation):
            self._opts.punctuation = punctuation
        if is_given(speech_rejection):
            self._opts.speech_rejection = speech_rejection

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
        registry: DeviceRegistry,
        app_key: str,
        base_url: str,
        reset_device_on_failure: bool,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._opts = opts
        self._session = session
        self._registry = registry
        self._app_key = app_key
        self._base_url = base_url
        self._reset_device_on_failure = reset_device_on_failure
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
        self._request_id = str(uuid.uuid4())
        device = await self._registry.get()
        query = urlencode({"aid": "401734", "device_id": device.device_id})
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    f"{self._base_url}?{query}",
                    headers={
                        "User-Agent": USER_AGENT,
                        "proto-version": "v2",
                        "x-custom-keepalive": "true",
                    },
                    heartbeat=30,
                    autoping=True,
                ),
                timeout=self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Doubao STT connection timed out") from None
        except aiohttp.ClientResponseError as error:
            raise APIStatusError(
                message=error.message,
                status_code=error.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as error:
            raise APIConnectionError("Failed to connect to Doubao STT") from error

        transcript = Transcript()
        finish_sent = asyncio.Event()
        audio_frames = 0
        try:
            try:
                await ws.send_bytes(
                    encode_request(
                        token=self._app_key, method="StartTask", request_id=self._request_id
                    )
                )
                await self._expect(ws, "TaskStarted")
                await ws.send_bytes(
                    encode_request(
                        token=self._app_key,
                        method="StartSession",
                        payload=self._session_config(device.device_id),
                        request_id=self._request_id,
                    )
                )
                await self._expect(ws, "SessionStarted")
            except Exception:
                if self._reset_device_on_failure:
                    await self._registry.reset()
                raise

            async def send() -> None:
                nonlocal audio_frames
                encoder = _OpusEncoder()
                chunker = utils.audio.AudioByteStream(
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                started_ms = int(time.time() * 1000)

                def metadata() -> str:
                    return json.dumps({"extra": {}, "timestamp_ms": started_ms + audio_frames * 20})

                async def send_packet(packet: bytes) -> None:
                    nonlocal audio_frames
                    await ws.send_bytes(
                        encode_request(
                            method="TaskRequest",
                            payload=metadata(),
                            audio=packet,
                            request_id=self._request_id,
                            frame=FIRST_FRAME if audio_frames == 0 else MIDDLE_FRAME,
                        )
                    )
                    audio_frames += 1
                    self._input_consumed = True

                try:
                    async for data in self._input_ch:
                        if isinstance(data, self._FlushSentinel):
                            continue
                        self._input_consumed = True
                        mono = _to_mono(data)
                        self._report_usage(mono.duration)
                        for frame in chunker.push(mono.data.tobytes()):
                            for packet in encoder.encode(frame):
                                await send_packet(packet)
                    for frame in chunker.flush():
                        for packet in encoder.encode(_pad_frame(frame, SAMPLES_PER_FRAME)):
                            await send_packet(packet)
                    await ws.send_bytes(
                        encode_request(
                            method="TaskRequest",
                            payload=metadata(),
                            request_id=self._request_id,
                            frame=LAST_FRAME,
                        )
                    )
                    finish_sent.set()
                    await ws.send_bytes(
                        encode_request(
                            token=self._app_key,
                            method="FinishSession",
                            request_id=self._request_id,
                        )
                    )
                finally:
                    self._flush_usage()

            async def receive() -> None:
                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        if finish_sent.is_set():
                            self._emit_updates(transcript.finish())
                            return
                        raise APIConnectionError(
                            "Doubao closed the STT websocket",
                            retryable=not self._input_consumed,
                        )
                    if msg.type is aiohttp.WSMsgType.ERROR:
                        raise APIConnectionError(
                            "Doubao STT websocket failed",
                            retryable=not self._input_consumed,
                        )
                    if msg.type is not aiohttp.WSMsgType.BINARY:
                        continue
                    response = Response.parse(msg.data)
                    if response.event.endswith("Failed"):
                        raise APIStatusError(
                            f"Doubao {response.event} {response.status}: {response.message}",
                            status_code=500,
                            request_id=self._request_id,
                            retryable=False,
                        )
                    if response.result:
                        self._mark_provider_progress()
                        self._emit_updates(transcript.update(response.result))
                    if response.event == "SessionFinished":
                        self._emit_updates(transcript.finish())
                        return

            send_task = asyncio.create_task(send(), name="doubao-stt-send")
            receive_task = asyncio.create_task(receive(), name="doubao-stt-receive")
            try:
                done, _ = await asyncio.wait(
                    (send_task, receive_task), return_when=asyncio.FIRST_COMPLETED
                )
                if send_task in done:
                    send_task.result()
                    timeout = 10.0 + audio_frames / 200.0
                    try:
                        await asyncio.wait_for(receive_task, timeout=timeout)
                    except asyncio.TimeoutError:
                        if not transcript.has_text:
                            raise APITimeoutError(
                                "Doubao STT final transcript timed out", retryable=False
                            ) from None
                        self._emit_updates(transcript.finish())
                else:
                    receive_task.result()
                    if not finish_sent.is_set():
                        raise APIConnectionError(
                            "Doubao STT session finished before input ended", retryable=False
                        )
            finally:
                await utils.aio.gracefully_cancel(send_task, receive_task)
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except aiohttp.ClientError as error:
            raise APIConnectionError(
                "Doubao STT connection failed", retryable=not self._input_consumed
            ) from error
        except Exception as error:
            raise APIConnectionError(
                "Doubao STT stream failed", retryable=not self._input_consumed
            ) from error
        finally:
            self._flush_usage()
            await ws.close()

    async def _expect(self, ws: aiohttp.ClientWebSocketResponse, event: str) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=self._conn_options.timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError(f"Timed out waiting for Doubao {event}") from None
            if msg.type is not aiohttp.WSMsgType.BINARY:
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    raise APIConnectionError(f"Doubao disconnected before {event}")
                continue
            response = Response.parse(msg.data)
            if response.event.endswith("Failed") or (
                response.event == event and response.status != OK_STATUS
            ):
                raise APIStatusError(
                    f"Doubao {response.event} {response.status}: {response.message}",
                    status_code=500,
                    request_id=self._request_id,
                )
            if response.event == event:
                return

    def _session_config(self, device_id: str) -> str:
        return json.dumps(
            {
                "audio_info": {
                    "channel": 1,
                    "format": "speech_opus",
                    "sample_rate": SAMPLE_RATE,
                },
                "enable_punctuation": self._opts.punctuation,
                "enable_speech_rejection": self._opts.speech_rejection,
                "extra": {
                    "app_name": "oime",
                    "cell_compress_rate": 8,
                    "did": device_id,
                    "enable_asr_threepass": False,
                    "enable_asr_twopass": True,
                    "input_mode": "stream",
                },
            },
            separators=(",", ":"),
        )

    def _emit_updates(self, updates: list[TranscriptUpdate]) -> None:
        end_of_speech = False
        for update in updates:
            if not update.text:
                continue
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            if update.final or self._opts.interim_results:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT
                        if update.final
                        else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            stt.SpeechData(
                                language=LanguageCode(self._opts.language), text=update.text
                            )
                        ],
                    )
                )
            end_of_speech = update.end_of_speech
        if end_of_speech and self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

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


def _join_texts(texts: list[str]) -> str:
    output = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if output and not _cjk(output[-1]) and not _cjk(text[0]):
            output += " "
        output += text
    return output


def _cjk(char: str) -> bool:
    code = ord(char)
    return 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0xFF00 <= code <= 0xFFEF
