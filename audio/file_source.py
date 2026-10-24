"""Replay a WAV/FLAC file as if it were a live audio source.

Emits the same `AudioChunk`s as live capture, paced in real time by default,
so the whole pipeline (segmenter -> STT -> translation -> UI) can be tested
repeatably without microphones, loopback or a call.

    src = FileSource("data/call1.wav", "system", chunk_queue, clock)
    src.start()          # background thread
    src.done.wait()      # set when the file (plus trailing silence) is sent

Use one FileSource per side, e.g. --mic-file me.wav --system-file others.wav.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

from core.types import SAMPLE_RATE, AudioChunk, Source

CHUNK_SAMPLES = 512  # 32 ms at 16 kHz, same size as a live capture chunk


def load_audio(path: str) -> np.ndarray:
    """Load any soundfile-readable file as 16 kHz mono float32."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(audio, dtype=np.float32)


class FileSource:
    """Background thread that streams a file into `out_queue` as `AudioChunk`s."""

    def __init__(
        self,
        path: str,
        source: Source,
        out_queue: "queue.Queue[AudioChunk]",
        clock: Callable[[], float],
        realtime: bool = True,
        start_offset: float = 0.0,
        trailing_silence: float = 1.5,
    ) -> None:
        """
        realtime:         pace chunks at real speed (False = as fast as possible,
                          handy for unit tests; latency numbers are then meaningless).
        start_offset:     seconds to wait before the file starts (to stagger two sides).
        trailing_silence: seconds of zeros appended so the last phrase gets closed.
        """
        self.path = path
        self.source = source
        self.out_queue = out_queue
        self.clock = clock
        self.realtime = realtime
        self.start_offset = start_offset

        audio = load_audio(path)
        pad = np.zeros(int(trailing_silence * SAMPLE_RATE), dtype=np.float32)
        self.audio = np.concatenate([audio, pad])
        self.duration = len(audio) / SAMPLE_RATE

        self.done = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"FileSource-{source}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            if self.start_offset and self._stop.wait(self.start_offset):
                return
            t0 = self.clock()  # session time of the file's first sample
            for i in range(0, len(self.audio), CHUNK_SAMPLES):
                if self._stop.is_set():
                    return
                t_chunk = t0 + i / SAMPLE_RATE
                if self.realtime:
                    # A live callback fires when a chunk has been fully *recorded*,
                    # so release each chunk at the time its last sample "happened".
                    t_release = t_chunk + CHUNK_SAMPLES / SAMPLE_RATE
                    delay = t_release - self.clock()
                    if delay > 0:
                        time.sleep(delay)
                pcm = self.audio[i : i + CHUNK_SAMPLES]
                self.out_queue.put(AudioChunk(self.source, t_chunk, pcm))
        finally:
            self.done.set()

    def __repr__(self) -> str:
        return f"FileSource({self.source}: {self.path}, {self.duration:.1f} s)"
