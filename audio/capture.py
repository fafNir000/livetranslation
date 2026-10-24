"""Live audio capture on Windows: microphone + system output (WASAPI loopback).

Uses PyAudioWPatch, a PortAudio fork that exposes every output device as a
"loopback" input device. No virtual cable is needed and the user keeps hearing
the call normally.

Each source runs its own PortAudio stream. Every callback converts the device
audio (often 48 kHz stereo int16) to 16 kHz mono float32 and puts an
`AudioChunk` on a shared queue.

Standalone use (milestone 1 audio spike):
    python -m audio.capture --list
    python -m audio.capture --record 10 --out recordings/
    python -m audio.capture --mic 1 --loopback 17 --record 10

Known quirk: a WASAPI loopback stream delivers NO callbacks while nothing is
playing. Chunk timestamps therefore come from the session clock, not a sample
counter, and the segmenter closes open segments on an idle timeout (see
segmenter.py) instead of waiting for silence frames that never arrive.
"""

from __future__ import annotations

import argparse
import queue
import sys
from typing import Callable

import numpy as np
import soxr

from core.types import SAMPLE_RATE, AudioChunk, SessionClock, Source

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover - only available on Windows
    pyaudio = None

CHUNK_MS = 32  # ~ one Silero VAD frame at 16 kHz (512 samples)


def _require_pyaudio() -> None:
    if pyaudio is None:
        sys.exit(
            "PyAudioWPatch is not installed (Windows only). "
            "Install it with `pip install PyAudioWPatch`, or use --input-file replay mode."
        )


# --------------------------------------------------------------------------- #
# Device discovery
# --------------------------------------------------------------------------- #
def list_devices() -> None:
    """Print WASAPI microphones and loopback devices with their indices."""
    _require_pyaudio()
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = wasapi.get("defaultInputDevice", -1)
        try:
            default_loop = p.get_default_wasapi_loopback()["index"]
        except (OSError, LookupError):
            default_loop = -1

        mics, loops = [], []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] != wasapi["index"] or d["maxInputChannels"] < 1:
                continue
            (loops if d.get("isLoopbackDevice") else mics).append(d)

        def show(title: str, devs: list[dict], default: int) -> None:
            print(f"\n{title}")
            for d in devs:
                mark = "*" if d["index"] == default else " "
                print(
                    f" {mark} [{d['index']:>3}] {d['name']}  "
                    f"({int(d['defaultSampleRate'])} Hz, {d['maxInputChannels']} ch)"
                )

        show("Microphones (use with --mic):", mics, default_mic)
        show("System audio / loopback (use with --loopback):", loops, default_loop)
        print("\n* = default device\n")
    finally:
        p.terminate()


def default_mic_index(p) -> int:
    idx = p.get_host_api_info_by_type(pyaudio.paWASAPI).get("defaultInputDevice", -1)
    if idx < 0:
        raise RuntimeError("No default WASAPI microphone found. Pick one with --mic.")
    return idx


def default_loopback_index(p) -> int:
    try:
        return p.get_default_wasapi_loopback()["index"]
    except (OSError, LookupError) as e:
        raise RuntimeError("No WASAPI loopback device found. Pick one with --loopback.") from e


# --------------------------------------------------------------------------- #
# Capture stream
# --------------------------------------------------------------------------- #
class AudioCapture:
    """Captures one device and emits 16 kHz mono `AudioChunk`s to `out_queue`."""

    def __init__(
        self,
        pa,  # shared pyaudio.PyAudio instance
        device_index: int,
        source: Source,
        out_queue: "queue.Queue[AudioChunk]",
        clock: Callable[[], float],
    ) -> None:
        self.pa = pa
        self.source = source
        self.out_queue = out_queue
        self.clock = clock

        info = pa.get_device_info_by_index(device_index)
        self.device_index = device_index
        self.device_name: str = info["name"]
        self.in_rate = int(info["defaultSampleRate"])
        self.channels = max(1, int(info["maxInputChannels"]))
        self.frames_per_buffer = int(self.in_rate * CHUNK_MS / 1000)

        self._resampler = soxr.ResampleStream(
            self.in_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
        )
        self._stream = None
        self.overflows = 0  # how often PortAudio reported dropped input

    # PortAudio calls this from its own thread; keep it short and never block.
    def _callback(self, in_data, frame_count, time_info, status):
        if status:
            self.overflows += 1
        t_first = self.clock() - frame_count / self.in_rate

        x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1)
        y = self._resampler.resample_chunk(x)

        if len(y):
            self.out_queue.put(AudioChunk(self.source, t_first, y.astype(np.float32, copy=False)))
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.in_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def __repr__(self) -> str:
        return (
            f"AudioCapture({self.source}: [{self.device_index}] {self.device_name}, "
            f"{self.in_rate} Hz, {self.channels} ch)"
        )


def open_captures(
    out_queue: "queue.Queue[AudioChunk]",
    clock: Callable[[], float],
    mic_index: int | None = None,
    loopback_index: int | None = None,
):
    """Create mic + loopback captures (defaults if indices are None).

    Returns (pa, [captures]). Call `.start()` on each capture, and on shutdown
    `.stop()` on each followed by `pa.terminate()`.
    """
    _require_pyaudio()
    pa = pyaudio.PyAudio()
    mic_index = default_mic_index(pa) if mic_index is None else mic_index
    loopback_index = default_loopback_index(pa) if loopback_index is None else loopback_index
    caps = [
        AudioCapture(pa, mic_index, "mic", out_queue, clock),
        AudioCapture(pa, loopback_index, "system", out_queue, clock),
    ]
    return pa, caps


# --------------------------------------------------------------------------- #
# Standalone spike: list devices / record both sources to WAV
# --------------------------------------------------------------------------- #
def _record(seconds: float, out_dir: str, mic: int | None, loopback: int | None) -> None:
    import os

    import soundfile as sf

    q: "queue.Queue[AudioChunk]" = queue.Queue()
    clock = SessionClock()
    pa, caps = open_captures(q, clock, mic, loopback)
    for c in caps:
        print(c)

    buffers: dict[str, list[np.ndarray]] = {"mic": [], "system": []}
    last_print = 0.0
    window: dict[str, list[np.ndarray]] = {"mic": [], "system": []}

    for c in caps:
        c.start()
    print(f"Recording {seconds:.0f} s ... (play something and speak into the mic)")
    try:
        while clock() < seconds:
            try:
                ch = q.get(timeout=0.1)
                buffers[ch.source].append(ch.pcm)
                window[ch.source].append(ch.pcm)
            except queue.Empty:
                pass
            if clock() - last_print >= 1.0:  # simple level meter, once per second
                last_print = clock()
                levels = []
                for s in ("mic", "system"):
                    w = np.concatenate(window[s]) if window[s] else np.zeros(1, np.float32)
                    rms_db = 20 * np.log10(np.sqrt(np.mean(w**2)) + 1e-9)
                    levels.append(f"{s}: {rms_db:6.1f} dBFS")
                    window[s] = []
                print("  " + " | ".join(levels))
    finally:
        for c in caps:
            c.stop()
        pa.terminate()

    os.makedirs(out_dir, exist_ok=True)
    for s, chunks in buffers.items():
        path = os.path.join(out_dir, f"{s}.wav")
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        sf.write(path, audio, SAMPLE_RATE)
        print(f"Wrote {path}  ({len(audio) / SAMPLE_RATE:.1f} s)")
    for c in caps:
        if c.overflows:
            print(f"Warning: {c.source} reported {c.overflows} input overflows")
    print(
        "Note: system.wav is shorter than the recording if nothing was playing "
        "(WASAPI loopback sends no data during silence)."
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Audio capture spike (mic + WASAPI loopback)")
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    ap.add_argument("--mic", type=int, default=None, help="microphone device index")
    ap.add_argument("--loopback", type=int, default=None, help="loopback device index")
    ap.add_argument("--record", type=float, metavar="SECONDS", help="record both sources to WAV")
    ap.add_argument("--out", default="recordings", help="output folder for --record")
    args = ap.parse_args(argv)

    if args.list or args.record is None:
        list_devices()
    if args.record:
        _record(args.record, args.out, args.mic, args.loopback)


if __name__ == "__main__":
    main()
