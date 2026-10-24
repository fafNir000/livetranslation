"""Segmenter state-machine tests using a fake energy VAD (no torch needed).

Run:  python -m pytest tests/
"""

import numpy as np

from audio.segmenter import FRAME_SAMPLES, Segmenter, VadParams
from core.types import SAMPLE_RATE, AudioChunk


class EnergyVAD:
    """Fake VAD: prob 1.0 for loud frames, 0.0 for quiet ones."""

    def __call__(self, frame):
        return 1.0 if np.abs(frame).mean() > 0.05 else 0.0

    def reset(self):
        pass


def tone(seconds):
    return (0.3 * np.ones(int(seconds * SAMPLE_RATE))).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed(seg, audio, t0=0.0, chunk=FRAME_SAMPLES):
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.process(AudioChunk("system", t0 + i / SAMPLE_RATE, audio[i : i + chunk]))
    return out


def test_single_phrase_closed_by_pause():
    s = Segmenter("system", EnergyVAD())
    segs = feed(s, np.concatenate([silence(1), tone(2), silence(1)]))
    assert len(segs) == 1
    seg = segs[0]
    assert abs(seg.t_end - 3.0) < 0.05  # speech ended at 3.0 s
    assert abs(seg.t_start - (1.0 - 0.2)) < 0.05  # 200 ms pre-roll pad
    assert not seg.forced_cut


def test_two_phrases():
    s = Segmenter("system", EnergyVAD())
    segs = feed(s, np.concatenate([tone(1), silence(1), tone(1), silence(1)]))
    assert len(segs) == 2


def test_short_blip_dropped():
    s = Segmenter("system", EnergyVAD())
    segs = feed(s, np.concatenate([silence(0.5), tone(0.1), silence(1)]))
    assert segs == []


def test_long_phrase_force_cut():
    s = Segmenter("system", EnergyVAD(), VadParams(max_segment_s=8.0))
    segs = feed(s, np.concatenate([tone(20), silence(1)]))
    assert len(segs) >= 3
    assert all(x.duration <= 8.1 for x in segs)
    assert segs[0].forced_cut and not segs[-1].forced_cut
    total = sum(x.duration for x in segs)
    assert abs(total - 20.2) < 0.3  # nothing lost, only trailing pad added


def test_odd_chunk_sizes():
    s = Segmenter("system", EnergyVAD())
    segs = feed(s, np.concatenate([silence(1), tone(2), silence(1)]), chunk=1411)
    assert len(segs) == 1


def test_idle_timeout_closes_phrase():
    """Loopback stops sending audio mid-phrase: idle check must close it."""
    s = Segmenter("system", EnergyVAD())
    assert feed(s, tone(2)) == []  # phrase open, no pause seen
    assert s.flush_if_idle(now=2.3) == []  # not idle long enough yet
    segs = s.flush_if_idle(now=2.7)
    assert len(segs) == 1 and abs(segs[0].t_end - 2.0) < 0.05


def test_gap_in_stream_closes_phrase():
    s = Segmenter("system", EnergyVAD())
    feed(s, tone(2))
    segs = feed(s, tone(1), t0=5.0)  # chunk timestamps jump 3 s
    assert len(segs) == 1
    assert len(s.flush()) == 1  # second phrase closed at end of stream
