"""Shared data types passed between pipeline stages.

Every stage talks to the next one only through these objects, so a model or
audio backend can be swapped without touching the rest of the pipeline.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# All audio inside the pipeline is 16 kHz mono float32 in [-1, 1].
SAMPLE_RATE = 16_000

Source = Literal["mic", "system"]


class SessionClock:
    """Seconds since the session started (monotonic, unaffected by wall-clock changes)."""

    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def __call__(self) -> float:
        return time.monotonic() - self._t0


@dataclass(frozen=True)
class AudioChunk:
    """A short block of audio from one source."""

    source: Source
    t: float  # session time (s) of the FIRST sample in pcm
    pcm: np.ndarray  # float32, mono, SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.pcm) / SAMPLE_RATE


_segment_ids = itertools.count(1)


@dataclass
class Segment:
    """One speech phrase cut out by the segmenter, ready for STT / speaker ID."""

    source: Source
    t_start: float  # session time (s) speech started
    t_end: float  # session time (s) speech ended -> latency is measured from here
    pcm: np.ndarray  # float32, mono, SAMPLE_RATE
    t_emitted: float = 0.0  # session time the segmenter released it
    forced_cut: bool = False  # True if cut at max length instead of at a pause
    id: int = field(default_factory=lambda: next(_segment_ids))

    @property
    def duration(self) -> float:
        return len(self.pcm) / SAMPLE_RATE
