"""Voice-activity segmentation: turns a stream of AudioChunks into speech Segments.

One `Segmenter` per source (mic / system), because each needs its own VAD state.
`SegmenterWorker` is the thread that reads the shared chunk queue, routes chunks
to the right segmenter and puts finished `Segment`s on the next queue.

How a segment is closed (the latency-critical part):
  1. Normal:     ~min_silence_ms of non-speech after speech  -> close at the pause.
  2. Max length: phrase reaches max_segment_s -> cut at the quietest frame in the
                 last ~1.5 s (less chance of cutting mid-word), and carry on.
  3. Idle/gap:   no audio arrives for idle_timeout_s while a phrase is open.
                 This happens with WASAPI loopback, which stops sending data when
                 the call goes silent, so the silence frames from (1) never come.

The VAD is passed in (anything with `__call__(frame) -> prob` and `reset()`), so
the state machine can be unit-tested without torch, using a fake VAD.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from core.types import SAMPLE_RATE, AudioChunk, Segment, Source

FRAME_SAMPLES = 512  # Silero VAD needs exactly 512 samples per call at 16 kHz
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE  # 0.032 s


class VAD(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroVAD:
    """Silero VAD (torch). One instance per source: the model is stateful."""

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self.model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame), SAMPLE_RATE))

    def reset(self) -> None:
        self.model.reset_states()


@dataclass
class VadParams:
    threshold: float = 0.5  # prob >= threshold starts speech
    neg_threshold: float = 0.35  # prob < neg_threshold counts as silence (hysteresis)
    min_silence_ms: int = 500  # pause length that ends a phrase
    speech_pad_ms: int = 200  # audio kept before onset / after offset
    min_speech_ms: int = 400  # shorter blips (coughs, clicks) are dropped
    max_segment_s: float = 8.0  # force a cut so long phrases don't break the 5 s budget
    cut_search_s: float = 1.5  # where to look for the quietest frame on a forced cut
    idle_timeout_s: float = 0.6  # close an open phrase if audio stops arriving
    gap_s: float = 0.25  # jump in chunk timestamps treated as a stream gap


class Segmenter:
    """VAD state machine for ONE audio source."""

    def __init__(self, source: Source, vad: VAD, params: VadParams | None = None) -> None:
        self.source = source
        self.vad = vad
        self.p = params or VadParams()

        self._pad_frames = max(1, round(self.p.speech_pad_ms / 1000 / FRAME_S))
        self._silence_frames_to_close = max(1, round(self.p.min_silence_ms / 1000 / FRAME_S))

        # Incoming audio not yet consumed as whole 512-sample frames.
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_t: float | None = None  # session time of _buf[0]

        self._preroll: deque[tuple[float, np.ndarray, float]] = deque(maxlen=self._pad_frames)
        self._reset_phrase()

    # ------------------------------------------------------------------ API
    def process(self, chunk: AudioChunk) -> list[Segment]:
        """Feed one chunk; returns any segments that were closed."""
        out: list[Segment] = []
        expected = self._expected_t()

        if expected is None or chunk.t - expected > self.p.gap_s:
            # First chunk, or the stream paused (e.g. loopback during silence).
            if self._in_speech:
                out += self._close()
            self._buf = chunk.pcm.astype(np.float32, copy=False)
            self._buf_t = chunk.t
            self._preroll.clear()
            self.vad.reset()
        else:
            self._buf = np.concatenate([self._buf, chunk.pcm])

        while len(self._buf) >= FRAME_SAMPLES:
            frame = self._buf[:FRAME_SAMPLES]
            t = self._buf_t
            self._buf = self._buf[FRAME_SAMPLES:]
            self._buf_t += FRAME_S
            out += self._step(frame, t)
        return out

    def flush_if_idle(self, now: float) -> list[Segment]:
        """Close an open phrase if no audio has arrived for idle_timeout_s."""
        expected = self._expected_t()
        if self._in_speech and expected is not None and now - expected > self.p.idle_timeout_s:
            return self._close()
        return []

    def flush(self) -> list[Segment]:
        """Close whatever is open (end of stream / shutdown)."""
        return self._close() if self._in_speech else []

    # ------------------------------------------------------------ internals
    def _expected_t(self) -> float | None:
        if self._buf_t is None:
            return None
        return self._buf_t + len(self._buf) / SAMPLE_RATE

    def _reset_phrase(self) -> None:
        self._in_speech = False
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._t_start = 0.0
        self._last_speech_end = 0.0
        self._silence_run = 0
        self._speech_frames = 0

    def _step(self, frame: np.ndarray, t: float) -> list[Segment]:
        prob = self.vad(frame)
        frame_end = t + FRAME_S

        if not self._in_speech:
            if prob >= self.p.threshold:
                self._in_speech = True
                self._t_start = self._preroll[0][0] if self._preroll else t
                self._frames = [f for _, f, _ in self._preroll] + [frame]
                self._probs = [p for _, _, p in self._preroll] + [prob]
                self._preroll.clear()
                self._speech_frames = 1
                self._silence_run = 0
                self._last_speech_end = frame_end
            else:
                self._preroll.append((t, frame, prob))
            return []

        self._frames.append(frame)
        self._probs.append(prob)
        if prob >= self.p.neg_threshold:
            self._silence_run = 0
            self._speech_frames += 1
            self._last_speech_end = frame_end
        else:
            self._silence_run += 1

        if self._silence_run >= self._silence_frames_to_close:
            return self._close()
        if frame_end - self._t_start >= self.p.max_segment_s:
            return self._forced_cut()
        return []

    def _close(self) -> list[Segment]:
        """Emit the current phrase (trimming trailing silence beyond the pad)."""
        drop = max(0, self._silence_run - self._pad_frames)
        frames = self._frames[: len(self._frames) - drop] if drop else self._frames
        seg = self._make_segment(frames, self._t_start, self._last_speech_end, forced=False)
        self._reset_phrase()
        return [seg] if seg else []

    def _forced_cut(self) -> list[Segment]:
        """Phrase too long: cut at the quietest recent frame and keep going."""
        n = len(self._frames)
        search = min(n - 1, max(1, round(self.p.cut_search_s / FRAME_S)))
        window = self._probs[n - search :]
        cut = n - search + int(np.argmin(window)) + 1  # cut AFTER the quietest frame

        head_frames, tail_frames = self._frames[:cut], self._frames[cut:]
        tail_probs = self._probs[cut:]
        t_cut = self._t_start + cut * FRAME_S
        seg = self._make_segment(head_frames, self._t_start, t_cut, forced=True)

        # Continue the phrase with the remaining frames.
        self._frames, self._probs = tail_frames, tail_probs
        self._t_start = t_cut
        self._speech_frames = sum(p >= self.p.neg_threshold for p in tail_probs)
        return [seg] if seg else []

    def _make_segment(
        self, frames: list[np.ndarray], t_start: float, t_end: float, forced: bool
    ) -> Segment | None:
        speech_s = self._speech_frames * FRAME_S
        if not frames or (speech_s * 1000 < self.p.min_speech_ms and not forced):
            return None
        return Segment(
            source=self.source,
            t_start=t_start,
            t_end=t_end,
            pcm=np.concatenate(frames),
            forced_cut=forced,
        )


class SegmenterWorker:
    """Thread: chunk queue -> per-source Segmenter -> segment queue."""

    def __init__(
        self,
        in_queue: "queue.Queue[AudioChunk]",
        out_queue: "queue.Queue[Segment]",
        clock: Callable[[], float],
        vad_factory: Callable[[], VAD] = SileroVAD,
        params: VadParams | None = None,
        sources: tuple[Source, ...] = ("mic", "system"),
    ) -> None:
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.clock = clock
        self.segmenters = {s: Segmenter(s, vad_factory(), params) for s in sources}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Segmenter", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and flush any open phrases."""
        self._stop.set()
        self._thread.join(timeout=2)

    def _emit(self, segments: list[Segment]) -> None:
        for seg in segments:
            seg.t_emitted = self.clock()
            self.out_queue.put(seg)

    def _run(self) -> None:
        last_idle_check = 0.0
        while not self._stop.is_set():
            try:
                chunk = self.in_queue.get(timeout=0.05)
                self._emit(self.segmenters[chunk.source].process(chunk))
            except queue.Empty:
                pass
            # The queue is shared, so a busy mic keeps get() from ever timing out.
            # Check idle sources on a timer rather than only on queue.Empty.
            now = self.clock()
            if now - last_idle_check >= 0.05:
                last_idle_check = now
                for seg in self.segmenters.values():
                    self._emit(seg.flush_if_idle(now))
        # Drain anything left and close open phrases.
        while True:
            try:
                chunk = self.in_queue.get_nowait()
            except queue.Empty:
                break
            self._emit(self.segmenters[chunk.source].process(chunk))
        for seg in self.segmenters.values():
            self._emit(seg.flush())
