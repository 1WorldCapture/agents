from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiohttp
import av
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils
from livekit.plugins.doubao import STT
from livekit.plugins.doubao._device import DeviceRegistry
from livekit.plugins.doubao._proto import (
    LAST_FRAME,
    OK_STATUS,
    Response,
    Transcript,
    _key,
    _read_varint,
    _string_field,
    _varint,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
async def _http_context():
    async with utils.http_context.open():
        yield


def _response(
    event: str, *, result: dict[str, Any] | None = None, status: int = OK_STATUS
) -> bytes:
    return (
        _string_field(4, event)
        + _key(5, 0)
        + _varint(status)
        + _string_field(7, json.dumps(result) if result is not None else "")
    )


def _request_fields(data: bytes) -> dict[int, int | bytes]:
    fields: dict[int, int | bytes] = {}
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 2:
            size, offset = _read_varint(data, offset)
            value = data[offset : offset + size]
            offset += size
        else:
            raise AssertionError(f"unexpected request wire type {wire_type}")
        fields[field] = value
    return fields


def _method(data: bytes) -> str:
    value = _request_fields(data)[5]
    assert isinstance(value, bytes)
    return value.decode()


class _FakeDoubaoServer:
    def __init__(self) -> None:
        self.methods: list[str] = []
        self.frame_markers: list[int] = []
        self.opus_packets: list[bytes] = []
        self.session_config: dict[str, Any] = {}
        self.device_id = ""

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.device_id = request.query.get("device_id", "")
        assert request.headers["proto-version"] == "v2"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.BINARY:
                continue
            fields = _request_fields(msg.data)
            method = fields[5]
            assert isinstance(method, bytes)
            method_text = method.decode()
            self.methods.append(method_text)
            if method_text == "StartTask":
                await ws.send_bytes(_response("TaskStarted"))
            elif method_text == "StartSession":
                payload = fields[6]
                assert isinstance(payload, bytes)
                self.session_config = json.loads(payload)
                await ws.send_bytes(_response("SessionStarted"))
            elif method_text == "TaskRequest":
                marker = fields.get(9, 0)
                assert isinstance(marker, int)
                self.frame_markers.append(marker)
                audio = fields.get(7)
                if isinstance(audio, bytes):
                    self.opus_packets.append(audio)
            elif method_text == "FinishSession":
                await ws.send_bytes(
                    _response(
                        "SessionResult",
                        result={
                            "results": [
                                {"text": "你好", "index": 0, "is_interim": True},
                            ]
                        },
                    )
                )
                await ws.send_bytes(
                    _response(
                        "SessionResult",
                        result={
                            "results": [
                                {
                                    "text": "你好，世界",
                                    "index": 0,
                                    "is_interim": False,
                                    "is_vad_finished": True,
                                }
                            ]
                        },
                    )
                )
                await ws.send_bytes(_response("SessionFinished"))
        return ws


async def _start_server(server: _FakeDoubaoServer) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}/ws"


async def test_streaming_protocol_and_opus_audio() -> None:
    server = _FakeDoubaoServer()
    runner, url = await _start_server(server)
    try:
        provider = STT(device_id="test-device", base_url=url)
        stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
        stream.push_frame(
            rtc.AudioFrame(
                data=b"\x00\x00" * 480 * 2,
                sample_rate=48000,
                num_channels=2,
                samples_per_channel=480,
            )
        )
        stream.end_input()
        events = [event async for event in stream]
        await stream.aclose()

        assert server.methods[:2] == ["StartTask", "StartSession"]
        assert server.methods[-1] == "FinishSession"
        assert server.frame_markers[-1] == LAST_FRAME
        assert server.device_id == "test-device"
        assert server.session_config["audio_info"] == {
            "channel": 1,
            "format": "speech_opus",
            "sample_rate": 16000,
        }
        assert server.opus_packets

        decoder = av.CodecContext.create("opus", "r")
        decoder.sample_rate = 16000
        decoder.open()
        decoded = [
            frame
            for packet_data in server.opus_packets
            for frame in decoder.decode(av.Packet(packet_data))
        ]
        assert len(server.opus_packets) == 1
        assert sum(frame.samples / frame.sample_rate for frame in decoded) == pytest.approx(0.02)

        interim = [
            event.alternatives[0].text
            for event in events
            if event.type is stt.SpeechEventType.INTERIM_TRANSCRIPT
        ]
        final = [
            event.alternatives[0].text
            for event in events
            if event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert interim == ["你好"]
        assert final == ["你好，世界"]
        speech_events = [
            event for event in events if event.type is not stt.SpeechEventType.RECOGNITION_USAGE
        ]
        assert speech_events[0].type is stt.SpeechEventType.START_OF_SPEECH
        assert speech_events[-1].type is stt.SpeechEventType.END_OF_SPEECH
    finally:
        await runner.cleanup()


async def test_recognize_uses_streaming_session() -> None:
    server = _FakeDoubaoServer()
    runner, url = await _start_server(server)
    try:
        provider = STT(device_id="test-device", base_url=url)
        event = await provider.recognize(
            rtc.AudioFrame(
                data=b"\x00\x00" * 320,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=320,
            ),
            conn_options=APIConnectOptions(max_retry=0),
        )
        assert event.alternatives[0].text == "你好，世界"
    finally:
        await runner.cleanup()


def test_transcript_preserves_final_sentences_and_current_interim() -> None:
    transcript = Transcript()
    updates = transcript.update(
        json.dumps(
            {
                "results": [
                    {
                        "text": "第一句",
                        "index": 0,
                        "is_interim": False,
                        "is_vad_finished": True,
                    },
                    {"text": "还没定稿", "index": 1, "is_interim": True},
                ]
            }
        )
    )
    assert [(item.text, item.final) for item in updates] == [("第一句", True), ("还没定稿", False)]
    assert (
        transcript.update(
            json.dumps(
                {
                    "results": [
                        {
                            "text": "第一句",
                            "index": 0,
                            "is_interim": False,
                            "is_vad_finished": True,
                        }
                    ]
                }
            )
        )
        == []
    )
    assert [item.text for item in transcript.finish()] == ["还没定稿"]


def test_cumulative_response_is_finalized_at_end_of_input() -> None:
    transcript = Transcript()
    updates = transcript.update('{"results":[{"text":"cumulative text"}]}')
    assert [(item.text, item.final) for item in updates] == [("cumulative text", False)]
    updates = transcript.finish()
    assert [(item.text, item.final) for item in updates] == [("cumulative text", True)]


def test_cumulative_duplicate_final_is_suppressed() -> None:
    transcript = Transcript()
    raw = '{"results":[{"text":"hello","extra":{"nonstream_result":true}}]}'
    assert len(transcript.update(raw)) == 1
    assert transcript.update(raw) == []
    assert transcript.update('{"results":[{"text":"hello","is_interim":true}]}') == []
    assert transcript.finish() == []
    assert len(transcript.update('{"results":[{"text":""}],"extra":{"vad_start":true}}')) == 0
    assert len(transcript.update(raw)) == 1


def test_response_parses_signed_status() -> None:
    response = Response.parse(_response("SessionFailed", status=(1 << 64) - 1))
    assert response.status == -1


async def test_device_registration_and_cache(tmp_path: Path) -> None:
    async def register(request: web.Request) -> web.Response:
        assert request.query["aid"] == "401734"
        body = await request.json()
        assert body["magic_tag"] == "ss_app_log"
        return web.json_response({"device_id_str": "registered-device"})

    app = web.Application()
    app.router.add_post("/register", register)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    cache_path = tmp_path / "device.json"
    try:
        async with aiohttp.ClientSession() as session:
            registry = DeviceRegistry(
                session=session,
                cache_path=cache_path,
                register_url=f"http://127.0.0.1:{port}/register",
            )
            device = await registry.get()
            assert device.device_id == "registered-device"
            assert json.loads(cache_path.read_text())["device_id"] == "registered-device"
            await registry.reset()
            assert not cache_path.exists()
    finally:
        await runner.cleanup()
