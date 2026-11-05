from __future__ import annotations

import hashlib
import json
from pathlib import Path

import aiohttp
import av
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils
from livekit.plugins.weixin import STT
from livekit.plugins.weixin._client import _BOOT_KEY, _CMD_DH, _CMD_UIN, _CMD_VOICE, _SIGN_KEY
from livekit.plugins.weixin._crypto import (
    aes_decrypt,
    aes_encrypt,
    public_key,
    shared_x,
    snappy_compress,
    snappy_decompress,
)
from livekit.plugins.weixin._proto import (
    Fields,
    encode_bytes_field,
    encode_string_field,
    encode_varint,
    encode_varint_field,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
async def _http_context():
    async with utils.http_context.open():
        yield


class _FakeWeixinServer:
    def __init__(self, *, polish_on_end: bool = False) -> None:
        self.private = 0x123456789ABCDEF
        self.public = public_key(self.private)
        self.session_key: bytes | None = None
        self.polish_on_end = polish_on_end
        self.paths: list[str] = []
        self.need_uin: list[bool] = []
        self.voice_payloads: list[bytes] = []
        self.voice_sequences: list[int] = []
        self.voice_totals: list[int] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        assert "wxws_pb" in request.headers.get("Sec-WebSocket-Protocol", "")
        ws = web.WebSocketResponse(protocols=("wxws_pb",))
        await ws.prepare(request)
        self.session_key = None
        async for message in ws:
            if message.type is not aiohttp.WSMsgType.BINARY:
                continue
            outer = Fields.parse(message.data)
            http = Fields.parse(outer.bytes_field(5) or b"")
            path = http.string(3) or ""
            body = http.bytes_field(6) or b""
            headers = self._headers(http)
            cmd = int(headers.get("Kb-CmdId", outer.varint(3, 0) or 0))
            self.paths.append(path)
            self._verify_request(headers, body, cmd)

            if cmd == _CMD_DH:
                response_body = self._exchange_key(body, headers)
            elif cmd == _CMD_UIN:
                assert self.session_key is not None
                Fields.parse(aes_decrypt(self.session_key, body))
                response_body = aes_encrypt(self.session_key, encode_varint_field(2, 123456))
            elif cmd == _CMD_VOICE:
                response_body = await self._voice(body, headers)
            elif cmd == 0:
                response_body = b""
            else:
                assert self.session_key is not None
                assert aes_decrypt(self.session_key, body) == b""
                response_body = b""
            await ws.send_bytes(_response_frame(response_body))
        return ws

    def _exchange_key(self, body: bytes, headers: dict[str, str]) -> bytes:
        request = Fields.parse(aes_decrypt(_BOOT_KEY, body))
        client_public = request.string(1)
        assert client_public
        self.session_key = shared_x(self.private, client_public)
        need_uin = request.varint(3) == 1
        self.need_uin.append(need_uin)
        response = encode_string_field(2, self.public)
        if need_uin:
            response += encode_string_field(4, "test-uin-token")
        else:
            assert request.varint(5) == 123456
        return aes_encrypt(_BOOT_KEY, response)

    async def _voice(self, body: bytes, headers: dict[str, str]) -> bytes:
        assert self.session_key is not None
        assert headers["Kb-CompressionType"] == "2"
        plain = snappy_decompress(aes_decrypt(self.session_key, body))
        inner = Fields.parse(Fields.parse(plain).bytes_field(1) or b"")
        payload = inner.bytes_field(4)
        sequence = inner.varint(7, 0) or 0
        is_end = inner.varint(6) == 1
        if payload:
            self.voice_payloads.append(payload)
        self.voice_sequences.append(sequence)
        self.voice_totals.append(inner.varint(11, 0) or 0)

        text = ""
        polished = ""
        if sequence == 1 and not is_end:
            text = "你好"
        elif is_end and payload:
            text = "你好，世界"
            if self.polish_on_end:
                polished = "你好，世界。"
        elif sequence == 0 and is_end:
            polished = "你好，世界。"
        result = encode_string_field(4, text) + encode_string_field(14, polished)
        response = encode_bytes_field(1, result)
        encrypted = aes_encrypt(self.session_key, snappy_compress(response))
        return encrypted

    @staticmethod
    def _headers(http: Fields) -> dict[str, str]:
        headers: dict[str, str] = {}
        for field in http:
            if field.number != 5 or not isinstance(field.value, bytes):
                continue
            header = Fields.parse(field.value)
            name, value = header.string(1), header.string(2)
            if name is not None and value is not None:
                headers[name] = value
        return headers

    @staticmethod
    def _verify_request(headers: dict[str, str], body: bytes, cmd: int) -> None:
        assert hashlib.md5(body).hexdigest().upper() == headers["Kb-BodyMd5"]
        timestamp = headers["Kb-TimeStamp"]
        trace = headers["Kb-TraceId"]
        if cmd == _CMD_DH:
            signed = ["5", "2.2.3(657)", "2", timestamp, headers["Kb-BodyMd5"], trace, str(cmd)]
        elif cmd == _CMD_UIN:
            signed = [
                "5",
                "2.2.3(657)",
                "2",
                timestamp,
                headers["Kb-BodyMd5"],
                headers["Kb-GenUinToken"],
                trace,
                str(cmd),
            ]
        else:
            signed = [
                "5",
                "2.2.3(657)",
                "2",
                str(cmd),
                "0",
                timestamp,
                headers["Kb-BodyMd5"],
                headers["Kb-Uin"],
                trace,
                headers["Kb-TaskId"],
            ]
        expected = hashlib.sha256(("".join(signed) + _SIGN_KEY).encode()).hexdigest().upper()
        assert headers["Kb-Sign"] == expected


def _response_frame(body: bytes) -> bytes:
    response = encode_varint_field(2, 200)
    if body:
        header = encode_string_field(1, "Kb-CompressionType") + encode_string_field(2, "2")
        response += encode_bytes_field(4, header)
    response += encode_bytes_field(5, body)
    return encode_bytes_field(4, response)


async def _start_server(server: _FakeWeixinServer) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}/ws"


def _audio_frame(*, samples: int = 6720) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=samples,
    )


def _decode_opus(payloads: list[bytes]) -> list[av.AudioFrame]:
    decoder = av.CodecContext.create("opus", "r")
    decoder.sample_rate = 16000
    decoder.open()
    decoded: list[av.AudioFrame] = []
    for index, payload in enumerate(payloads):
        if index == 0:
            assert payload.startswith(b"#!OPUS_RAW_V1\x02\x01\x00")
            payload = payload[16:]
        offset = 0
        while offset < len(payload):
            size = int.from_bytes(payload[offset : offset + 2], "little")
            offset += 2
            packet = payload[offset : offset + size]
            assert len(packet) == size
            decoded.extend(decoder.decode(av.Packet(packet)))
            offset += size
    return decoded


async def test_streaming_protocol_identity_cache_and_recognize(tmp_path: Path) -> None:
    server = _FakeWeixinServer()
    runner, url = await _start_server(server)
    cache_path = tmp_path / "identity.json"
    try:
        provider = STT(base_url=url, identity_cache_path=cache_path)
        stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
        stream.push_frame(_audio_frame())
        stream.end_input()
        events = [event async for event in stream]
        await stream.aclose()

        assert server.paths[:4] == ["/timestamp", "/oauth_pubkey_v2", "/gen_uin_v2", "/api_v2"]
        assert server.need_uin == [True]
        assert server.voice_sequences == [1, 2, 0]
        assert len(server.voice_payloads) == 2
        assert server.voice_totals[-1] == sum(map(len, server.voice_payloads))
        decoded = _decode_opus(server.voice_payloads)
        assert len(decoded) == 7
        assert sum(frame.samples / frame.sample_rate for frame in decoded) == pytest.approx(0.14)
        identity = json.loads(cache_path.read_text())
        assert identity["uin"] == "123456"
        assert identity["device"].startswith("MAC")

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
        assert interim == ["你好", "你好，世界"]
        assert final == ["你好，世界。"]
        speech_events = [
            event for event in events if event.type is not stt.SpeechEventType.RECOGNITION_USAGE
        ]
        assert speech_events[0].type is stt.SpeechEventType.START_OF_SPEECH
        assert speech_events[-1].type is stt.SpeechEventType.END_OF_SPEECH
        usage = next(
            event for event in events if event.type is stt.SpeechEventType.RECOGNITION_USAGE
        )
        assert usage.recognition_usage.audio_duration == pytest.approx(0.14)

        result = await provider.recognize(
            _audio_frame(samples=480), conn_options=APIConnectOptions(max_retry=0)
        )
        assert result.alternatives[0].text == "你好，世界。"
        assert server.need_uin == [True, False]
        await provider.aclose()
    finally:
        await runner.cleanup()


async def test_interim_results_can_be_disabled(tmp_path: Path) -> None:
    server = _FakeWeixinServer(polish_on_end=True)
    runner, url = await _start_server(server)
    try:
        provider = STT(
            base_url=url,
            identity_cache_path=tmp_path / "identity.json",
            interim_results=False,
        )
        event = await provider.recognize(
            _audio_frame(samples=480), conn_options=APIConnectOptions(max_retry=0)
        )
        assert event.alternatives[0].text == "你好，世界。"
        assert server.voice_sequences == [1]
        await provider.aclose()
    finally:
        await runner.cleanup()


def test_device_and_uin_must_be_provided_together() -> None:
    with pytest.raises(ValueError, match="provided together"):
        STT(device="device-only")


def test_snappy_literals_and_overlapping_copies() -> None:
    for value in (b"", b"hello", bytes(range(256)), "微信语音识别".encode()):
        assert snappy_decompress(snappy_compress(value)) == value

    compressed = encode_varint(12) + bytes(((4 - 1) << 2,)) + b"abcd"
    compressed += bytes(((8 - 1) << 2 | 2, 4, 0))
    assert snappy_decompress(compressed) == b"abcdabcdabcd"


def test_secp128r1_shared_secret_and_public_key_validation() -> None:
    first, second = 1, 2
    assert public_key(first).startswith("04")
    assert shared_x(first, public_key(second)) == shared_x(second, public_key(first))
    assert len(shared_x(first, public_key(second))) == 32
    with pytest.raises(ValueError, match="public key"):
        shared_x(first, "04" + "00" * 64)


def test_aes_256_ecb_pkcs7_roundtrip() -> None:
    key = bytes(range(32))
    for value in (b"", b"16-byte-message!", "微信".encode()):
        assert aes_decrypt(key, aes_encrypt(key, value)) == value


def test_protobuf_multiple_fields_and_truncation() -> None:
    encoded = (
        encode_varint_field(1, 300)
        + encode_string_field(2, "first")
        + encode_string_field(2, "last")
        + encode_bytes_field(3, b"\x00\x01")
    )
    fields = Fields.parse(encoded)
    assert fields.varint(1) == 300
    assert fields.string(2) == "last"
    assert fields.bytes_field(3) == b"\x00\x01"
    with pytest.raises(ValueError, match="truncated"):
        Fields.parse(encoded[:-1])
