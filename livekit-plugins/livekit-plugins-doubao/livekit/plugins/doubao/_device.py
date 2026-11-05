from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError

DEFAULT_APP_KEY = "OrnqKvSSrs"
DEFAULT_DEVICE_CACHE_PATH = Path.home() / ".livekit" / "doubao-device.json"
DEVICE_REGISTER_URL = "https://log-klink.zijieapi.com/service/2/device_register/"
USER_AGENT = (
    "com.bytedance.android.doubaoime/100406010 "
    "(Linux; U; Android 16; en_US; Pixel 7 Pro; Build/BP2A.250605.031.A2; "
    "Cronet/TTNetVersion:94cf429a 2025-11-17 QuicVersion:1f89f732 2025-05-08)"
)

_APP_PARAMS = {
    "aid": "401734",
    "app_name": "oime",
    "channel": "official",
    "version_code": "100406010",
    "version_name": "1.4.6",
    "manifest_version_code": "100406010",
    "update_version_code": "100406010",
    "package": "com.bytedance.android.doubaoime",
}
_DEVICE_PARAMS = {
    "device_platform": "android",
    "os": "android",
    "os_api": "36",
    "os_version": "16",
    "device_type": "Pixel 7 Pro",
    "device_brand": "google",
    "device_model": "Pixel 7 Pro",
    "resolution": "1080*2340",
    "dpi": "420",
    "language": "zh",
    "timezone": "8",
    "access": "wifi",
    "rom": "BP2A.250605.031.A2",
    "rom_version": "BP2A.250605.031.A2",
}


@dataclass(frozen=True)
class Device:
    device_id: str


class DeviceRegistry:
    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        device_id: str | None = None,
        cache_path: str | Path | None = None,
        register_url: str = DEVICE_REGISTER_URL,
        timeout: float = 10.0,
    ) -> None:
        self._session = session
        self._device = Device(device_id) if device_id else None
        self._cache_path = Path(cache_path) if cache_path else DEFAULT_DEVICE_CACHE_PATH
        self._register_url = register_url
        self._timeout = timeout
        self._lock = asyncio.Lock()

    async def get(self) -> Device:
        if self._device:
            return self._device
        async with self._lock:
            if self._device:
                return self._device
            cached = await asyncio.to_thread(self._read_cache)
            if cached:
                self._device = cached
                return cached
            self._device = await self._register()
            await asyncio.to_thread(self._write_cache, self._device)
            return self._device

    async def reset(self) -> None:
        async with self._lock:
            self._device = None
            await asyncio.to_thread(self._cache_path.unlink, missing_ok=True)

    def _read_cache(self) -> Device | None:
        try:
            data = json.loads(self._cache_path.read_text())
            device_id = data.get("device_id")
            if isinstance(device_id, str) and device_id and device_id != "0":
                return Device(device_id)
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _write_cache(self, device: Device) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._cache_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"device_id": device.device_id}))
            temporary.replace(self._cache_path)
        except OSError:
            pass

    async def _register(self) -> Device:
        client_id = uuid.uuid4()
        now_ms = int(time.time() * 1000)
        query = {
            **_APP_PARAMS,
            **_DEVICE_PARAMS,
            "ssmix": "a",
            "_rticket": str(now_ms),
            "cdid": str(client_id),
            "ac": "wifi",
        }
        header: dict[str, Any] = {
            **query,
            "device_id": 0,
            "install_id": 0,
            "cdid": str(client_id),
            "openudid": f"{secrets.randbits(64):016x}",
            "clientudid": str(uuid.uuid4()),
            "region": "CN",
            "tz_name": "Asia/Shanghai",
            "tz_offset": 28800,
            "sim_region": "cn",
            "carrier_region": "cn",
            "cpu_abi": "arm64-v8a",
            "build_serial": "unknown",
            "not_request_sender": 0,
            "sig_hash": "",
            "google_aid": "",
            "mc": "",
            "serial_number": "",
        }
        body = {"magic_tag": "ss_app_log", "header": header, "_gen_time": now_ms}
        try:
            async with self._session.post(
                self._register_url,
                params=query,
                json=body,
                headers={"User-Agent": USER_AGENT},
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status != 200:
                    raise APIStatusError(
                        "Doubao device registration failed",
                        status_code=response.status,
                        request_id=None,
                        body=None,
                    )
                data = await response.json()
        except (APIStatusError, APITimeoutError):
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError("Doubao device registration timed out") from None
        except Exception as error:
            raise APIConnectionError("Doubao device registration failed") from error

        if not isinstance(data, dict):
            raise APIStatusError(
                "Doubao device registration returned an invalid response",
                status_code=200,
                request_id=None,
                retryable=False,
            )
        device_id = data.get("device_id_str") or data.get("device_id")
        if not isinstance(device_id, (str, int)) or not str(device_id) or str(device_id) == "0":
            raise APIStatusError(
                "Doubao device registration was rejected",
                status_code=200,
                request_id=None,
                retryable=False,
            )
        return Device(str(device_id))
