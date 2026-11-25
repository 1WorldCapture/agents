from __future__ import annotations

import asyncio
import os
import re
import shutil
import ssl
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import certifi
import numpy as np
from kokoro_onnx import Kokoro
from misaki import zh

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_VOICE = "zf_001"
MODEL_FILE = "kokoro-v1.1-zh.onnx"
VOICES_FILE = "voices-v1.1-zh.bin"
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
# Characters per clause below which a clause joins the next one, and the silence between clauses.
MIN_CLAUSE_LENGTH = 8
CLAUSE_PAUSE = 0.1

# Loading, G2P, and inference all run on this one thread. ONNX Runtime already spreads one
# request over every core, and jieba, which the Chinese G2P uses, is not thread-safe.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")


def default_model_dir() -> Path:
    """The directory that holds the model files: ``KOKORO_MODEL_DIR``, else ``~/.cache/kokoro``."""
    return Path(os.environ.get("KOKORO_MODEL_DIR") or Path.home() / ".cache" / "kokoro")


def download_model(directory: Path | None = None) -> Path:
    """Download the Kokoro v1.1-zh model and voices into ``directory`` unless they are there."""
    directory = directory or default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    for name in (MODEL_FILE, VOICES_FILE):
        path = directory / name
        if path.exists():
            continue
        logger.info("downloading %s to %s", name, directory)
        partial = path.with_name(name + ".part")
        with urllib.request.urlopen(f"{RELEASE_URL}/{name}", context=context) as response:
            with partial.open("wb") as file:
                shutil.copyfileobj(response, file)
        # Rename only a complete file, so an interrupted download starts over next time.
        partial.replace(path)
    return directory


class _Engine:
    def __init__(self, directory: Path) -> None:
        self.kokoro = Kokoro(str(directory / MODEL_FILE), str(directory / VOICES_FILE))
        self._g2p = zh.ZHG2P(  # type: ignore[no-untyped-call]
            version="1.1",
            en_callable=lambda en: self.kokoro.tokenizer.phonemize(en, "en-us"),
        )
        # The first call builds jieba's dictionary; do it now instead of on the first reply.
        self._g2p("你好")

    def phonemize(self, text: str) -> str:
        phonemes, _ = self._g2p(text)
        return str(phonemes)


def _clauses(text: str) -> list[str]:
    """Split a sentence after each comma, semicolon, and colon.

    Each clause goes through G2P and the model on its own and plays as soon as it is ready, so
    a long sentence starts playing after its first clause instead of after the whole sentence.
    An ASCII mark splits only before a space, so ``1,000`` and ``10:30`` stay whole. A clause
    shorter than ``MIN_CLAUSE_LENGTH`` characters joins the next one, since very short pieces
    sound choppy.
    """
    clauses: list[str] = []
    pending = ""
    for piece in re.split(r"(?<=[，；：])|(?<=[,;:])(?=\s)", text):
        pending += piece
        if len(pending.strip()) >= MIN_CLAUSE_LENGTH:
            clauses.append(pending.strip())
            pending = ""
    if clauses and pending.strip():
        clauses[-1] += pending.rstrip()
    elif pending.strip():
        clauses.append(pending.strip())
    return clauses


@cache
def _engine(directory: Path) -> _Engine:
    return _Engine(download_model(directory))


@dataclass
class _TTSOptions:
    voice: str
    speed: float


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        model_dir: str | Path | None = None,
    ) -> None:
        """Create a local Kokoro text-to-speech instance for Chinese, with inline English.

        It runs Kokoro-82M v1.1-zh with ONNX Runtime on the CPU. The model files (about 380 MB)
        are downloaded on first use, or ahead of time with ``python agent.py download-files``.

        Args:
            voice: Voice name, ``zf_001`` to ``zf_099`` (female) or ``zm_009`` to ``zm_100``
                (male). Defaults to ``zf_001``.
            speed: Speech rate from 0.5 to 2.0. Defaults to 1.0.
            model_dir: Directory for the model files. Defaults to ``KOKORO_MODEL_DIR``, else
                ``~/.cache/kokoro``.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(voice=voice, speed=speed)
        self._model_dir = Path(model_dir) if model_dir else default_model_dir()

    @property
    def model(self) -> str:
        return "kokoro-v1.1-zh"

    @property
    def provider(self) -> str:
        return "Kokoro"

    def prewarm(self) -> None:
        """Start loading the model in the background, so the first reply does not wait for it."""
        _executor.submit(_engine, self._model_dir)

    async def _load(self) -> _Engine:
        return await asyncio.get_running_loop().run_in_executor(_executor, _engine, self._model_dir)

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
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update options used by synthesis requests started after this call."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    async def list_voices(self) -> list[str]:
        """Return the voice names of the model, loading it if needed."""
        return (await self._load()).kokoro.get_voices()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        try:
            engine = await self._tts._load()
            spoken = False
            for clause in _clauses(self.input_text):
                phonemes = await asyncio.get_running_loop().run_in_executor(
                    _executor, engine.phonemize, clause
                )
                # Kokoro makes noise for phonemes that are only punctuation; skip them.
                if not any(char.isalpha() for char in phonemes):
                    continue
                if spoken:
                    output_emitter.push(bytes(int(CLAUSE_PAUSE * SAMPLE_RATE) * 2))
                spoken = True
                async for samples, _ in engine.kokoro.create_stream(
                    phonemes, voice=self._opts.voice, speed=self._opts.speed, is_phonemes=True
                ):
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                    output_emitter.push(pcm.tobytes())
            # Text with nothing to say still needs some audio, or the stream counts as failed.
            if not spoken:
                output_emitter.push(bytes(SAMPLE_RATE // 20 * 2))
            output_emitter.flush()
        except Exception as error:
            # A local model fails the same way on every attempt, so do not retry.
            raise APIConnectionError(f"Kokoro TTS failed: {error}", retryable=False) from error
