"""Live Armenian -> English transcription/translation (orchestrator).

Wires the finished stages together with threads and queues (PLAN.md §2):

    Mic (WASAPI) ──────┐
                       ├─► chunk_q ─► SegmenterWorker ─► seg_q ─► Pipeline worker ─► terminal
    Loopback (WASAPI) ─┘                                          │  STT (Whisper)
                                                                  │  speaker ID (pyannote, runs
                                                                  │     in parallel with STT)
                                                                  └  MT (NLLB)

Usage:
    python main.py --list-devices
    python main.py                                  # default mic + default loopback
    python main.py --mic 1 --loopback 17
    python main.py --system-file recordings/other/system.wav   # replay mode, no live audio
    python main.py --debug                          # per-stage timings + dropped segments

Output (one block per phrase, printed as soon as it is ready):
    [00:01:12] Speaker 1
    HY: Բարև, ինչպե՞ս ես։
    EN: Hi, how are you?
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

from core.types import AudioChunk, Segment, SessionClock

LATENCY_BUDGET_S = 5.0
DEFAULT_STT_MODEL = "models/whisper-turbo-hy"
DEFAULT_MT_MODEL = "models/nllb-200-distilled-1.3B"

log = logging.getLogger("main")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def fmt_time(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


class Printer:
    """Prints result blocks; a lock keeps blocks from interleaving."""

    def __init__(self, debug: bool) -> None:
        self.debug = debug
        self._lock = threading.Lock()

    def block(self, t: float, label: str, hy: str, en: str, timing: str = "") -> None:
        lines = [f"[{fmt_time(t)}] {label}", f"HY: {hy}", f"EN: {en}"]
        if self.debug and timing:
            lines.append(f"    {timing}")
        with self._lock:
            print("\n".join(lines), flush=True)

    def info(self, msg: str) -> None:
        with self._lock:
            print(msg, file=sys.stderr, flush=True)


class LatencyStats:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.dropped = 0

    def summary(self) -> str:
        if not self.values:
            return f"No phrases printed ({self.dropped} dropped by the STT filter)."
        v = sorted(self.values)
        p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
        over = sum(x > LATENCY_BUDGET_S for x in v)
        return (
            f"{len(v)} phrase(s) printed, {self.dropped} dropped. Latency (end of phrase -> EN): "
            f"mean {sum(v) / len(v):.2f} s, p95 {p95:.2f} s, max {v[-1]:.2f} s, "
            f"{over} over {LATENCY_BUDGET_S:.0f} s."
        )


# --------------------------------------------------------------------------- #
# Pipeline worker: Segment -> (STT || speaker ID) -> MT -> print
# --------------------------------------------------------------------------- #
class PipelineWorker:
    """Single GPU worker (PLAN.md §2). Segments are handled in arrival order.

    Speaker embedding runs on its own single thread so it overlaps with
    Whisper; using one thread also keeps PyannoteSpeakerID's clustering
    state single-threaded as it requires.
    """

    def __init__(self, seg_q: "queue.Queue[Segment | None]", stt, mt, speaker_id,
                 clock, printer: Printer, stats: LatencyStats) -> None:
        self.seg_q = seg_q
        self.stt = stt
        self.mt = mt
        self.speaker_id = speaker_id
        self.clock = clock
        self.printer = printer
        self.stats = stats
        self._spk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SpeakerID")
        self._thread = threading.Thread(target=self._run, name="Pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        self._spk_pool.shutdown(wait=False, cancel_futures=True)

    def _identify(self, seg: Segment) -> tuple[str, float]:
        t0 = time.perf_counter()
        label = self.speaker_id(seg)
        return label, time.perf_counter() - t0

    def _run(self) -> None:
        while True:
            seg = self.seg_q.get()
            if seg is None:  # sentinel: shutdown after everything before it is done
                return
            try:
                self._process(seg)
            except Exception:  # one bad segment must never kill the live session
                log.exception("Segment %s failed", seg.id)

    def _process(self, seg: Segment) -> None:
        t_picked = self.clock()
        spk_future = self._spk_pool.submit(self._identify, seg)

        stt = self.stt.transcribe_segment(seg)
        if not stt.ok:
            self.stats.dropped += 1
            spk_future.cancel()
            if self.printer.debug:
                self.printer.info(
                    f"  [{fmt_time(seg.t_start)}] {seg.source} {seg.duration:.1f}s dropped: "
                    f"{stt.reason} (raw: {stt.raw_text!r})"
                )
            return

        mt = self.mt.translate_stt(stt)
        try:
            label, spk_s = spk_future.result()
        except Exception:
            log.exception("Speaker ID failed for segment %s", seg.id)
            label, spk_s = ("Me" if seg.source == "mic" else "Speaker ?"), 0.0

        en = mt.text if mt.ok else "(no translation)"
        latency = self.clock() - seg.t_end
        self.stats.values.append(latency)

        timing = (
            f"[latency {latency:.2f}s | queue {t_picked - seg.t_emitted:.2f}s "
            f"vad {seg.t_emitted - seg.t_end:.2f}s stt {stt.elapsed_s:.2f}s "
            f"mt {mt.elapsed_s:.2f}s spk {spk_s:.2f}s | audio {seg.duration:.1f}s"
            f"{' forced-cut' if seg.forced_cut else ''}"
            f"{' | MT low confidence: ' + mt.reason if mt.low_confidence else ''}]"
        )
        self.printer.block(seg.t_start, label, stt.text, en, timing)
        if latency > LATENCY_BUDGET_S:
            log.warning("Segment %s over budget: %.2f s", seg.id, latency)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live Armenian speech -> Armenian text + English translation, by speaker.")
    g = ap.add_argument_group("audio input")
    g.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    g.add_argument("--mic", type=int, default=None, help="microphone device index")
    g.add_argument("--loopback", type=int, default=None, help="system-audio loopback device index")
    g.add_argument("--mic-file", help="replay mode: WAV played as the microphone ('Me')")
    g.add_argument("--system-file", help="replay mode: WAV played as system audio (call side)")
    g.add_argument("--system-offset", type=float, default=0.0,
                   help="replay mode: seconds to delay --system-file (to stagger the two sides)")

    g = ap.add_argument_group("models")
    g.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    g.add_argument("--stt-model", default=None,
                   help=f"Whisper model name or CTranslate2 folder (default: {DEFAULT_STT_MODEL} "
                        "if present, else WhisperConfig default)")
    g.add_argument("--stt-compute-type", default=None)
    g.add_argument("--stt-beam-size", type=int, default=None)
    g.add_argument("--mt-model", default=DEFAULT_MT_MODEL)
    g.add_argument("--mt-compute-type", default=None)
    g.add_argument("--mt-beam-size", type=int, default=None)
    g.add_argument("--speaker-threshold", type=float, default=None,
                   help="cosine similarity needed to match a known speaker")
    g.add_argument("--speaker-device", default=None, help="device for pyannote (default: --device)")
    g.add_argument("--no-speaker-id", action="store_true",
                   help="skip pyannote: all system audio is labelled 'Speaker 1'")
    g.add_argument("--offline", action="store_true",
                   help="never touch the network (models must already be on disk)")

    ap.add_argument("--debug", action="store_true",
                    help="show per-stage timings, dropped segments and info logs")
    return ap.parse_args(argv)


def setup_console(debug: bool) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252/cp437
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not debug:
        warnings.filterwarnings("ignore")  # torch / pyannote deprecation noise
    try:
        from dotenv import load_dotenv

        load_dotenv()  # HF_TOKEN for pyannote
    except ImportError:
        pass


def load_models(args: argparse.Namespace, printer: Printer):
    """Load and warm up every model before audio starts (PLAN.md §2)."""
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch  # noqa: F401  (first, so CTranslate2 finds torch's CUDA DLLs on Windows)

    from stt.whisper import WhisperConfig, WhisperSTT
    from translate.nllb import NllbConfig, NllbTranslator

    wcfg = WhisperConfig(device=args.device, local_files_only=args.offline)
    if args.stt_model:
        wcfg.model = args.stt_model
    elif os.path.isdir(DEFAULT_STT_MODEL):
        wcfg.model = DEFAULT_STT_MODEL
    if args.stt_compute_type:
        wcfg.compute_type = args.stt_compute_type
    if args.stt_beam_size is not None:
        wcfg.beam_size = args.stt_beam_size

    t0 = time.perf_counter()
    printer.info(f"Loading STT {wcfg.model!r} ({wcfg.compute_type}, beam {wcfg.beam_size}) ...")
    stt = WhisperSTT(wcfg)

    ncfg = NllbConfig(model=args.mt_model, device=args.device)
    if args.mt_compute_type:
        ncfg.compute_type = args.mt_compute_type
    if args.mt_beam_size is not None:
        ncfg.beam_size = args.mt_beam_size
    printer.info(f"Loading translation {ncfg.model!r} ({ncfg.compute_type}, beam {ncfg.beam_size}) ...")
    mt = NllbTranslator(ncfg)

    if args.no_speaker_id:
        def speaker_id(seg: Segment) -> str:
            return "Me" if seg.source == "mic" else "Speaker 1"
    else:
        from speakers.pyannote import PyannoteSpeakerID, SpeakerParams

        params = SpeakerParams()
        if args.speaker_threshold is not None:
            params.match_threshold = args.speaker_threshold
        printer.info("Loading speaker model (pyannote) ...")
        try:
            sid = PyannoteSpeakerID(device=args.speaker_device or args.device, params=params)
        except Exception as e:
            sys.exit(f"Speaker model failed to load: {e}\n"
                     "Fix HF_TOKEN in .env, or run with --no-speaker-id.")
        sid.warmup()
        speaker_id = sid.identify

    printer.info(f"All models ready in {time.perf_counter() - t0:.1f} s.")
    return stt, mt, speaker_id


def start_sources(args, chunk_q, clock, printer: Printer):
    """Start file replay or live capture. Returns (stop_fn, replay_done_fn or None)."""
    if args.mic_file or args.system_file:
        from audio.file_source import FileSource

        sources = []
        if args.mic_file:
            sources.append(FileSource(args.mic_file, "mic", chunk_q, clock))
        if args.system_file:
            sources.append(FileSource(args.system_file, "system", chunk_q, clock,
                                      start_offset=args.system_offset))
        for s in sources:
            printer.info(f"  {s}")
            s.start()

        def stop() -> None:
            for s in sources:
                s.stop()

        return stop, (lambda: all(s.done.is_set() for s in sources))

    from audio.capture import open_captures

    pa, caps = open_captures(chunk_q, clock, args.mic, args.loopback)
    for c in caps:
        printer.info(f"  {c}")
        c.start()

    def stop() -> None:
        for c in caps:
            try:
                c.stop()
            except Exception:
                pass
        pa.terminate()
        for c in caps:
            if c.overflows:
                printer.info(f"Warning: {c.source} reported {c.overflows} input overflows")

    return stop, None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_console(args.debug)

    if args.list_devices:
        from audio.capture import list_devices

        list_devices()
        return

    printer = Printer(args.debug)
    stt, mt, speaker_id = load_models(args, printer)

    from audio.segmenter import SegmenterWorker

    chunk_q: "queue.Queue[AudioChunk]" = queue.Queue()
    seg_q: "queue.Queue[Segment | None]" = queue.Queue()
    clock = SessionClock()  # [00:00:00] = when listening starts
    stats = LatencyStats()

    segmenter = SegmenterWorker(chunk_q, seg_q, clock)
    worker = PipelineWorker(seg_q, stt, mt, speaker_id, clock, printer, stats)
    worker.start()
    segmenter.start()
    stop_sources, replay_done = start_sources(args, chunk_q, clock, printer)
    printer.info("Listening... press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.2)
            if replay_done is not None and replay_done():
                time.sleep(1.0)  # let the segmenter close the last phrase
                break
    except KeyboardInterrupt:
        printer.info("\nStopping...")
    finally:
        stop_sources()
        segmenter.stop()  # flushes open phrases into seg_q
        seg_q.put(None)  # worker finishes what is queued, then exits
        try:
            worker.join(timeout=30 if replay_done is not None else 10)
        except KeyboardInterrupt:
            pass
        printer.info(stats.summary())


if __name__ == "__main__":
    main()
