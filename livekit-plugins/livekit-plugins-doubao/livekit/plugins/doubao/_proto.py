from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

OK_STATUS = 20_000_000
FIRST_FRAME = 1
MIDDLE_FRAME = 3
LAST_FRAME = 9


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode a negative value")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _key(field: int, wire_type: int) -> bytes:
    return _varint(field << 3 | wire_type)


def _bytes_field(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _string_field(field: int, value: str) -> bytes:
    return _bytes_field(field, value.encode())


def encode_request(
    *,
    method: str,
    request_id: str,
    token: str = "",
    payload: str = "",
    audio: bytes = b"",
    frame: int = 0,
) -> bytes:
    request = bytearray()
    if token:
        request += _string_field(2, token)
    request += _string_field(3, "ASR")
    request += _string_field(5, method)
    if payload:
        request += _string_field(6, payload)
    if audio:
        request += _bytes_field(7, audio)
    request += _string_field(8, request_id)
    if frame:
        request += _key(9, 0) + _varint(frame)
    return bytes(request)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


@dataclass(frozen=True)
class Response:
    event: str = ""
    status: int = 0
    message: str = ""
    result: str = ""

    @classmethod
    def parse(cls, data: bytes) -> Response:
        fields: dict[int, list[int | bytes]] = {}
        offset = 0
        while offset < len(data):
            key, offset = _read_varint(data, offset)
            field, wire_type = key >> 3, key & 7
            if wire_type == 0:
                value: int | bytes
                value, offset = _read_varint(data, offset)
            elif wire_type == 1:
                end = offset + 8
                if end > len(data):
                    raise ValueError("truncated fixed64 field")
                value = data[offset:end]
                offset = end
            elif wire_type == 2:
                size, offset = _read_varint(data, offset)
                end = offset + size
                if end > len(data):
                    raise ValueError("truncated length-delimited field")
                value = data[offset:end]
                offset = end
            elif wire_type == 5:
                end = offset + 4
                if end > len(data):
                    raise ValueError("truncated fixed32 field")
                value = data[offset:end]
                offset = end
            else:
                raise ValueError(f"unsupported protobuf wire type: {wire_type}")
            fields.setdefault(field, []).append(value)

        def string(field: int) -> str:
            values = fields.get(field)
            if not values or not isinstance(values[-1], bytes):
                return ""
            return values[-1].decode(errors="replace")

        status_values = fields.get(5, [])
        raw_status = status_values[-1] if status_values else 0
        if not isinstance(raw_status, int):
            raw_status = 0
        raw_status &= 0xFFFFFFFF
        status = raw_status - (1 << 32) if raw_status & (1 << 31) else raw_status
        return cls(
            event=string(4),
            status=status,
            message=string(6),
            result=string(7),
        )


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    final: bool
    end_of_speech: bool


class Transcript:
    def __init__(self) -> None:
        self._final_indices: set[int] = set()
        self._current = ""
        self._next_index = 0
        self._last_cumulative_final: str | None = None
        self._last_final_text = ""
        self.has_text = False

    def update(self, raw_result: str) -> list[TranscriptUpdate]:
        try:
            result: Any = json.loads(raw_result)
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            return []

        results = [item for item in result["results"] if isinstance(item, dict)]
        if not results:
            return []
        extra = result.get("extra")
        if isinstance(extra, dict) and extra.get("vad_start") is True:
            self._last_cumulative_final = None

        if not any("index" in item for item in results):
            item = results[0]
            text = item.get("text")
            if not isinstance(text, str) or not text:
                return []
            final, end_of_speech = _final_flags(item)
            # After a final, the server repeats the same text, sometimes marked interim, until
            # the next utterance starts; treating it as new speech would send the turn twice.
            if text == self._last_cumulative_final:
                return []
            # The text covers the whole session, so later utterances start with the earlier
            # ones; report only the words after the last final.
            # A final equal to the last one after vad_start is the same words said again.
            session_text = text
            prefix = self._last_final_text
            if prefix and text.startswith(prefix) and not (final and text == prefix):
                text = text[len(prefix) :].lstrip()
                if not text:
                    return []
            if final:
                self._last_cumulative_final = session_text
                self._last_final_text = session_text
            self._current = "" if final else text
            self.has_text = True
            return [TranscriptUpdate(text, final, end_of_speech)]

        updates: list[TranscriptUpdate] = []
        for item in results:
            text = item.get("text")
            if not isinstance(text, str) or not text:
                continue
            self.has_text = True
            final, end_of_speech = _final_flags(item)
            if final:
                index = item.get("index")
                if not isinstance(index, int):
                    index = self._next_index
                self._next_index = max(self._next_index, index + 1)
                if index not in self._final_indices:
                    self._final_indices.add(index)
                    self._current = ""
                    updates.append(TranscriptUpdate(text, True, end_of_speech))
            else:
                self._current = text
                updates.append(TranscriptUpdate(text, False, False))
        return updates

    def finish(self) -> list[TranscriptUpdate]:
        if not self._current:
            return []
        text = self._current
        self._current = ""
        return [TranscriptUpdate(text, True, True)]


def _final_flags(item: dict[str, Any]) -> tuple[bool, bool]:
    extra = item.get("extra")
    nonstream = isinstance(extra, dict) and extra.get("nonstream_result") is True
    vad_finished = item.get("is_interim") is False and item.get("is_vad_finished") is True
    return nonstream or vad_finished, vad_finished or nonstream
