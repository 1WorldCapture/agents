from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint cannot encode a negative value")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
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


def encode_key(field: int, wire_type: int) -> bytes:
    return encode_varint(field << 3 | wire_type)


def encode_varint_field(field: int, value: int) -> bytes:
    return encode_key(field, 0) + encode_varint(value)


def encode_bytes_field(field: int, value: bytes) -> bytes:
    return encode_key(field, 2) + encode_varint(len(value)) + value


def encode_string_field(field: int, value: str) -> bytes:
    return encode_bytes_field(field, value.encode())


@dataclass(frozen=True)
class Field:
    number: int
    value: int | bytes


class Fields:
    def __init__(self, fields: Iterable[Field] = ()) -> None:
        self._fields = tuple(fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self._fields)

    def varint(self, number: int, default: int | None = None) -> int | None:
        for field in reversed(self._fields):
            if field.number == number and isinstance(field.value, int):
                return field.value
        return default

    def bytes_field(self, number: int, default: bytes | None = None) -> bytes | None:
        for field in reversed(self._fields):
            if field.number == number and isinstance(field.value, bytes):
                return field.value
        return default

    def string(self, number: int, default: str | None = None) -> str | None:
        value = self.bytes_field(number)
        if value is None:
            return default
        return value.decode(errors="replace")

    @classmethod
    def parse(cls, data: bytes) -> Fields:
        fields: list[Field] = []
        offset = 0
        while offset < len(data):
            key, offset = read_varint(data, offset)
            number, wire_type = key >> 3, key & 7
            if number == 0:
                raise ValueError("invalid protobuf field number")
            if wire_type == 0:
                value: int | bytes
                value, offset = read_varint(data, offset)
            elif wire_type == 1:
                end = offset + 8
                if end > len(data):
                    raise ValueError("truncated fixed64 field")
                value = data[offset:end]
                offset = end
            elif wire_type == 2:
                size, offset = read_varint(data, offset)
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
            fields.append(Field(number, value))
        return cls(fields)
