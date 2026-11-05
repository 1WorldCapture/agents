from __future__ import annotations

import secrets

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from ._proto import encode_varint, read_varint

_P = 0xFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFF
_A = 0xFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFC
_B = 0xE87579C11079F43DD824993C2CEE5ED3
_N = 0xFFFFFFFE0000000075A30D1B9038A115
_G = (
    0x161FF7528B899B2D0C28607CA52C5B86,
    0xCF5AC8395BAFEB13C02DA292DDED7A83,
)
_Point = tuple[int, int] | None


def aes_encrypt(key: bytes, data: bytes) -> bytes:
    _check_key(key)
    padder = PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES256(key), modes.ECB()).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def aes_decrypt(key: bytes, data: bytes) -> bytes:
    _check_key(key)
    decryptor = Cipher(algorithms.AES256(key), modes.ECB()).decryptor()
    padded = decryptor.update(data) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def _check_key(key: bytes) -> None:
    if len(key) != 32:
        raise ValueError("AES-256 requires a 32-byte key")


def snappy_compress(data: bytes) -> bytes:
    output = bytearray(encode_varint(len(data)))
    for offset in range(0, len(data), 60):
        chunk = data[offset : offset + 60]
        output.append((len(chunk) - 1) << 2)
        output.extend(chunk)
    return bytes(output)


def snappy_decompress(data: bytes) -> bytes:
    length, offset = read_varint(data, 0)
    output = bytearray()

    def read_byte() -> int:
        nonlocal offset
        if offset >= len(data):
            raise ValueError("truncated snappy data")
        value = data[offset]
        offset += 1
        return value

    def copy(copy_offset: int, count: int) -> None:
        if copy_offset <= 0 or copy_offset > len(output) or len(output) + count > length:
            raise ValueError("invalid snappy copy")
        for _ in range(count):
            output.append(output[-copy_offset])

    while offset < len(data):
        tag = read_byte()
        kind = tag & 3
        if kind == 0:
            count = tag >> 2
            if count >= 60:
                width = count - 59
                count = 0
                for index in range(width):
                    count |= read_byte() << (8 * index)
            count += 1
            if offset + count > len(data) or len(output) + count > length:
                raise ValueError("invalid snappy literal")
            output.extend(data[offset : offset + count])
            offset += count
        elif kind == 1:
            copy(((tag >> 5) << 8) | read_byte(), ((tag >> 2) & 7) + 4)
        elif kind == 2:
            copy(read_byte() | (read_byte() << 8), (tag >> 2) + 1)
        else:
            copy_offset = 0
            for index in range(4):
                copy_offset |= read_byte() << (8 * index)
            copy(copy_offset, (tag >> 2) + 1)
    if len(output) != length:
        raise ValueError("snappy length mismatch")
    return bytes(output)


def generate_keypair() -> tuple[int, str]:
    private = secrets.randbelow(_N - 1) + 1
    return private, public_key(private)


def public_key(private: int) -> str:
    if not 1 <= private < _N:
        raise ValueError("invalid secp128r1 private key")
    point = _multiply(private, _G)
    if point is None:
        raise ValueError("invalid secp128r1 private key")
    return f"04{point[0]:032X}{point[1]:032X}"


def shared_x(private: int, peer_public: str) -> bytes:
    point = _decode_public(peer_public)
    shared = _multiply(private, point)
    if shared is None:
        raise ValueError("invalid secp128r1 shared point")
    return f"{shared[0]:032X}".encode()


def _decode_public(value: str) -> tuple[int, int]:
    try:
        if len(value) != 66 or not value.startswith("04"):
            raise ValueError
        x, y = int(value[2:34], 16), int(value[34:], 16)
    except ValueError:
        raise ValueError("invalid secp128r1 public key") from None
    if not (0 <= x < _P and 0 <= y < _P) or (y * y - x * x * x - _A * x - _B) % _P:
        raise ValueError("invalid secp128r1 public key")
    return x, y


def _add(point: _Point, other: _Point) -> _Point:
    if point is None:
        return other
    if other is None:
        return point
    x1, y1 = point
    x2, y2 = other
    if x1 == x2 and (y1 + y2) % _P == 0:
        return None
    if point == other:
        if y1 == 0:
            return None
        slope = (3 * x1 * x1 + _A) * pow(2 * y1, -1, _P) % _P
    else:
        slope = (y2 - y1) * pow((x2 - x1) % _P, -1, _P) % _P
    x3 = (slope * slope - x1 - x2) % _P
    return x3, (slope * (x1 - x3) - y1) % _P


def _multiply(scalar: int, point: _Point) -> _Point:
    result: _Point = None
    addend = point
    while scalar:
        if scalar & 1:
            result = _add(result, addend)
        addend = _add(addend, addend)
        scalar >>= 1
    return result
