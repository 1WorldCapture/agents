from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import string
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError

from ._crypto import (
    aes_decrypt,
    aes_encrypt,
    generate_keypair,
    shared_x,
    snappy_compress,
    snappy_decompress,
)
from ._proto import (
    Fields,
    encode_bytes_field,
    encode_string_field,
    encode_varint_field,
)

DEFAULT_WS_URL = "wss://wetype.weixin.qq.com/"
DEFAULT_IDENTITY_CACHE_PATH = Path.home() / ".livekit" / "weixin-identity.json"
_HOST = "wetype.weixin.qq.com"
_SIGN_KEY = "zN7rB3bL4pO8jW1o"
_BOOT_KEY = b"D4Y5U3Y2M0C0T7N4P1P7O2N6E1I2Y1U6"
_VERSION = "2.2.3(657)"
_OS_TYPE = "5"
_PLATFORM = "2"
_DEVICE_MODEL = "Mac16,12"
_CMD_DH = 2_147_483_646
_CMD_UIN = 0x7FFF_FDFD
_CMD_NOTIFY = 8074
_CMD_VOICE = 4548


@dataclass(frozen=True)
class Identity:
    device: str
    uin: str


@dataclass(frozen=True)
class VoiceResult:
    text: str
    polished: str


@dataclass(frozen=True)
class _Reply:
    status: int
    headers: dict[str, str]
    body: bytes


class IdentityStore:
    def __init__(
        self,
        *,
        identity: Identity | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self._explicit = identity
        self._identity = identity
        self._cache_path = Path(cache_path) if cache_path else DEFAULT_IDENTITY_CACHE_PATH
        self._lock = asyncio.Lock()

    async def load(self) -> Identity | None:
        if self._identity:
            return self._identity
        async with self._lock:
            if not self._identity:
                self._identity = await asyncio.to_thread(self._read_cache)
            return self._identity

    async def save(self, identity: Identity) -> None:
        if self._explicit:
            return
        async with self._lock:
            self._identity = identity
            await asyncio.to_thread(self._write_cache, identity)

    def _read_cache(self) -> Identity | None:
        try:
            data = json.loads(self._cache_path.read_text())
            device, uin = data.get("device"), data.get("uin")
            if isinstance(device, str) and device and isinstance(uin, str) and uin and uin != "0":
                return Identity(device=device, uin=uin)
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _write_cache(self, identity: Identity) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_path.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
            temporary.write_text(json.dumps({"device": identity.device, "uin": identity.uin}))
            temporary.chmod(0o600)
            temporary.replace(self._cache_path)
        except OSError:
            pass


class WeTypeClient:
    def __init__(
        self,
        *,
        ws: aiohttp.ClientWebSocketResponse,
        store: IdentityStore,
        timeout: float,
    ) -> None:
        self._ws = ws
        self._store = store
        self._timeout = timeout
        self._task = 0
        self._uin = "0"
        self._device = ""
        self._session_key: bytes | None = None
        self._server_public = ""
        self._uin_token = ""

    @classmethod
    async def open(
        cls,
        *,
        session: aiohttp.ClientSession,
        base_url: str,
        store: IdentityStore,
        timeout: float,
    ) -> WeTypeClient:
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    base_url,
                    protocols=("wxws_pb",),
                    heartbeat=30,
                    autoping=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Weixin STT connection timed out") from None
        except aiohttp.ClientResponseError as error:
            raise APIStatusError(
                message=error.message,
                status_code=error.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as error:
            raise APIConnectionError("Failed to connect to Weixin STT") from error

        client = cls(ws=ws, store=store, timeout=timeout)
        try:
            await client._handshake()
            return client
        except Exception:
            await ws.close()
            raise

    async def aclose(self) -> None:
        await self._ws.close()

    async def voice(
        self,
        *,
        voice_id: str,
        opus: bytes | None,
        seq: int,
        total_bytes: int,
        is_end: bool,
    ) -> VoiceResult:
        inner = bytearray(encode_string_field(2, voice_id))
        if opus:
            inner += encode_bytes_field(4, opus)
        inner += encode_varint_field(5, 5)
        if is_end:
            inner += encode_varint_field(6, 1)
        inner += encode_varint_field(7, seq)
        if total_bytes > 0:
            inner += encode_varint_field(11, total_bytes)
        inner += encode_varint_field(22, 1)
        inner += encode_varint_field(23, 1)
        inner += encode_varint_field(24, 1)
        outer = encode_bytes_field(1, bytes(inner))

        body = aes_encrypt(self._key(), snappy_compress(outer))
        reply = await self._roundtrip("/api_v2", body, cmd=_CMD_VOICE, compress="2")
        if reply.status != 200 or not reply.body:
            raise APIStatusError(
                f"Weixin voice request failed with status {reply.status}",
                status_code=_status_code(reply.status),
                request_id=None,
                retryable=False,
            )
        plain = aes_decrypt(self._key(), reply.body)
        compression = reply.headers.get("Kb-CompressionType") or reply.headers.get(
            "CompressionType"
        )
        if compression == "2":
            plain = snappy_decompress(plain)
        top = Fields.parse(plain)
        result = Fields.parse(top.bytes_field(1) or plain)
        return VoiceResult(text=result.string(4, "") or "", polished=result.string(14, "") or "")

    async def _handshake(self) -> None:
        await self._roundtrip("/timestamp", b"", cmd=0)
        saved = await self._store.load()
        if saved:
            self._device, self._uin = saved.device, saved.uin
            try:
                await self._exchange_key(need_uin=False)
            except (APIConnectionError, APIStatusError, ValueError):
                await self._register()
        else:
            await self._register()
        await self._store.save(Identity(device=self._device, uin=self._uin))
        await self._roundtrip("/api_v2", aes_encrypt(self._key(), b""), cmd=_CMD_NOTIFY)

    async def _exchange_key(self, *, need_uin: bool) -> None:
        private, public = generate_keypair()
        request = bytearray(encode_string_field(1, public) + encode_string_field(2, public))
        if need_uin:
            request += encode_varint_field(3, 1)
        request += encode_string_field(4, self._device)
        if not need_uin:
            request += encode_varint_field(5, int(self._uin))
        request += encode_string_field(7, "")
        request += encode_string_field(8, _DEVICE_MODEL)
        request += encode_string_field(9, "")
        body = await self._request(
            "/oauth_pubkey_v2",
            aes_encrypt(_BOOT_KEY, bytes(request)),
            cmd=_CMD_DH,
        )
        response = Fields.parse(aes_decrypt(_BOOT_KEY, body))
        server_public = response.string(2)
        token = response.string(4)
        if not server_public or (need_uin and token is None):
            raise ValueError("incomplete Weixin key exchange response")
        self._server_public = server_public
        self._uin_token = token or ""
        self._session_key = shared_x(private, server_public)

    async def _register(self) -> None:
        self._uin = "0"
        body = "MAC" + "0" * (17 - len(_DEVICE_MODEL)) + _DEVICE_MODEL + _random_lowercase(12)
        digest = hashlib.md5((body + _SIGN_KEY).encode()).hexdigest().upper()
        self._device = body + digest
        await self._exchange_key(need_uin=True)
        request = (
            encode_string_field(1, self._device)
            + encode_string_field(2, self._server_public)
            + encode_string_field(4, self._uin_token)
        )
        response = await self._request(
            "/gen_uin_v2",
            aes_encrypt(self._key(), request),
            cmd=_CMD_UIN,
            token=self._uin_token,
        )
        issued = Fields.parse(aes_decrypt(self._key(), response)).varint(2)
        if not issued:
            raise APIStatusError(
                "Weixin did not issue a UIN",
                status_code=500,
                request_id=None,
                retryable=False,
            )
        self._uin = str(issued)

    def _key(self) -> bytes:
        if self._session_key is None:
            raise ValueError("missing Weixin session key")
        return self._session_key

    async def _request(self, path: str, body: bytes, *, cmd: int, token: str = "") -> bytes:
        reply = await self._roundtrip(path, body, cmd=cmd, token=token)
        if reply.status != 200 or not reply.body:
            raise APIStatusError(
                f"Weixin {path} failed with status {reply.status}",
                status_code=_status_code(reply.status),
                request_id=None,
            )
        return reply.body

    async def _roundtrip(
        self,
        path: str,
        body: bytes,
        *,
        cmd: int,
        compress: str = "1",
        token: str = "",
    ) -> _Reply:
        dh, gen_uin = cmd == _CMD_DH, cmd == _CMD_UIN
        if not (dh or gen_uin):
            self._task += 1
        task_id = cmd if dh or gen_uin else self._task
        trace = _random_lowercase(16)
        body_md5 = hashlib.md5(body).hexdigest().upper()
        timestamp = str(int(time.time() * 1000))
        if dh:
            signed = [_OS_TYPE, _VERSION, _PLATFORM, timestamp, body_md5, trace, str(cmd)]
        elif gen_uin:
            signed = [
                _OS_TYPE,
                _VERSION,
                _PLATFORM,
                timestamp,
                body_md5,
                token,
                trace,
                str(cmd),
            ]
        else:
            signed = [
                _OS_TYPE,
                _VERSION,
                _PLATFORM,
                str(cmd),
                "0",
                timestamp,
                body_md5,
                self._uin,
                trace,
                str(task_id),
            ]
        signature = hashlib.sha256(("".join(signed) + _SIGN_KEY).encode()).hexdigest().upper()

        headers: list[tuple[str, str]] = []
        if token:
            headers.append(("Kb-GenUinToken", token))
        headers.append(("Kb-Uin", self._uin))
        if dh or gen_uin:
            headers.append(("Kb-DeviceCodeRestrictionV2", "1"))
        if not dh:
            if self._session_key is not None:
                headers.append(
                    ("Kb-SharedKeySuffix", self._session_key[-4:].decode(errors="replace"))
                )
            headers.extend((("Kb-CmdId", str(cmd)), ("Kb-SubCmdId", "0")))
        headers.extend(
            (
                ("Kb-OsType", _OS_TYPE),
                ("Kb-Version", _VERSION),
                ("Kb-SystemVersion", "27.0.0"),
                ("Kb-PackageType", "3"),
                ("Use_DebugNet", "0"),
                ("Kb-TimeStamp", timestamp),
                ("Kb-BodyMd5", body_md5),
                ("Kb-TraceId", trace),
                ("Kb-TaskId", str(task_id)),
                ("Kb-Scene", "2"),
                ("Kb-Sign", signature),
                ("Content-Length", str(len(body))),
                ("Kb-CompressionType", compress),
                ("Content-Type", "application/octet-stream"),
                ("HOST", _HOST),
            )
        )

        http = bytearray(
            encode_string_field(1, "POST")
            + encode_string_field(3, path)
            + encode_string_field(4, "")
        )
        for name, value in headers:
            header = encode_string_field(1, name) + encode_string_field(2, value)
            http += encode_bytes_field(5, header)
        http += encode_bytes_field(6, body)
        frame = (
            encode_varint_field(1, 0)
            + encode_varint_field(2, 0)
            + encode_varint_field(3, task_id)
            + encode_bytes_field(5, bytes(http))
        )
        await self._ws.send_bytes(frame)

        while True:
            try:
                message = await asyncio.wait_for(self._ws.receive(), timeout=self._timeout)
            except asyncio.TimeoutError:
                raise APITimeoutError(f"Weixin {path} request timed out") from None
            if message.type is aiohttp.WSMsgType.BINARY:
                break
            if message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                raise APIConnectionError(f"Weixin disconnected during {path}")

        top = Fields.parse(message.data)
        payload = top.bytes_field(4)
        response = Fields.parse(payload) if payload is not None else Fields()
        response_headers: dict[str, str] = {}
        for field in response:
            if field.number == 5 or not isinstance(field.value, bytes):
                continue
            try:
                response_header = Fields.parse(field.value)
                header_name = response_header.bytes_field(1)
                header_value = response_header.bytes_field(2)
            except ValueError:
                continue
            if header_name is not None and header_value is not None:
                response_headers[header_name.decode(errors="replace")] = header_value.decode(
                    errors="replace"
                )
        return _Reply(
            status=response.varint(2, 0) or 0,
            headers=response_headers,
            body=response.bytes_field(5, b"") or b"",
        )


def _random_lowercase(length: int) -> str:
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def _status_code(status: int) -> int:
    return status if 100 <= status <= 599 else 500
