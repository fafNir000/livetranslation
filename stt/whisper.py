"""Armenian speech-to-text with faster-whisper (CTranslate2).

Input is one `Segment` from the segmenter (16 kHz mono float32, <= ~8 s).
Output is an `SttResult`: the Armenian text plus the confidence numbers we
use to throw away hallucinations, and the time the call took (for the 5 s
latency budget).

    stt = WhisperSTT(WhisperConfig(model="large-v3"))   # loads + warms up
    res = stt.transcribe_segment(segment)
    if res.ok:
        print(res.text)
    else:
        log.debug("dropped segment %s: %s", segment.id, res.reason)

Which model is used is only a config value, so the STT benchmark can swap
between the DECISION.md candidates without code changes:
  - "large-v3" / "large-v3-turbo"         (downloaded by faster-whisper)
  - a local CTranslate2 directory          (e.g. a converted BuzzASR/armenian or
                                            Chillarmo turbo-armenian fine-tune:
                                            ct2-transformers-converter --model <hf id>
                                            --output_dir models/<name>
                                            --copy_files tokenizer.json preprocessor_config.json
                                            --quantization float16)

Design choices (see PLAN.md §3, §5):
  * language is forced to "hy"; no auto-detect (it costs time and mislabels
    short Armenian phrases as Russian or Persian).
  * condition_on_previous_text=False and no timestamps: every segment is
    decoded on its own. That's faster and stops one bad segment from
    feeding a hallucination loop into the next.
  * faster-whisper's own VAD is off: our segmenter has already cut on speech.
  * Small temperature fallback (0.0 -> 0.2 -> 0.4) instead of Whisper's six
    steps, so a "hard" segment can't take several decode passes and blow the
    latency budget.
  * A post-filter (`check_hallucination`) rejects typical Whisper output on
    noise/silence: very low confidence, repetition loops, the YouTube
    "thanks for watching / subscribe" phrases, text in the wrong script, and
    far more text than the audio could hold.

Standalone check (no live audio needed):
    python -m stt.whisper recordings/mic.wav
    python -m stt.whisper recordings/system.wav --segment --model large-v3-turbo
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from core.types import SAMPLE_RATE, Segment

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config and result types
# --------------------------------------------------------------------------- #
@dataclass
class WhisperConfig:
    model: str = "large-v3"  # model name or path to a CTranslate2 model dir
    device: str = "cuda"  # "cuda" | "cpu" | "auto"
    device_index: int = 0
    # int8 weights: large-v3 needs ~1.6 GB instead of ~3.1 GB, so it fits a 4 GB
    # card (GTX 1650) without spilling into shared system RAM. GTX 16xx cards also
    # have no tensor cores, so float16 brings no speed-up there.
    compute_type: str = "int8_float32"
    download_root: str | None = None  # model cache dir (None = HF default cache)
    local_files_only: bool = False  # True for the interview: never touch the network

    # Decoding
    language: str = "hy"
    # GTX 1650 benchmark (bench/stt_speed.py): large-v3 int8_float32 takes 1.2 s/phrase
    # at beam 1 vs 1.4-1.5 s at beam 5, with near-identical text. Re-check per model.
    beam_size: int = 1
    # Single greedy/beam pass. Every fallback temperature is a FULL extra decode
    # (it doubled/tripled latency in the benchmark); a bad first pass is caught by
    # the hallucination filter instead.
    temperatures: tuple[float, ...] = (0.0,)
    # Cap on generated tokens per second of audio. Armenian takes many BPE tokens
    # per word, but a looping decoder ("կկկկկ...") otherwise runs to 448 tokens.
    max_tokens_per_s: float = 30.0
    initial_prompt: str | None = None  # e.g. names / terms expected in the call
    hotwords: str | None = None
    no_repeat_ngram_size: int = 0  # >0 hard-blocks repeats, but can hurt real speech

    # faster-whisper internal thresholds (per decoded sub-segment)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # Our post-filter (whole segment); tune on the benchmark clips
    min_avg_logprob: float = -1.2  # reject below this
    max_no_speech_prob: float = 0.8  # reject if the model is this sure it's not speech...
    no_speech_logprob: float = -0.6  # ...and confidence is below this
    max_compression_ratio: float = 2.4  # after collapsing repeats
    min_armenian_ratio: float = 0.6  # share of letters that must be Armenian script
    max_chars_per_s: float = 25.0  # fast Armenian speech is ~15 chars/s
    min_audio_s: float = 0.3  # shorter input isn't worth a GPU call

    # Input level: Whisper does worse on very quiet audio (a laptop mic often peaks
    # around -20 dBFS). Boost each segment to target_peak, but never by more than
    # max_gain, so near-silent noise isn't blown up into "speech".
    normalize_gain: bool = True
    target_peak: float = 0.9
    max_gain: float = 10.0  # +20 dB

    warmup: bool = True


@dataclass
class SttResult:
    """What the STT stage hands to translation (or to the debug log)."""

    text: str  # Armenian text ("" if rejected)
    ok: bool  # False -> don't translate or print it
    reason: str = ""  # why it was rejected (for --debug / benchmark logs)
    raw_text: str = ""  # decoder output before cleanup/filtering
    avg_logprob: float = 0.0  # duration-weighted over sub-segments
    no_speech_prob: float = 0.0  # max over sub-segments
    compression_ratio: float = 0.0  # of the cleaned text
    audio_s: float = 0.0  # length of the input audio
    elapsed_s: float = 0.0  # wall time spent in the model
    segment_id: int | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Text cleanup and hallucination filter (pure functions, unit-testable)
# --------------------------------------------------------------------------- #
# Whisper's training data is full of YouTube subtitles, so on silence or noise
# it likes to "hear" the credits. Two lists:
#   _HALLUCINATION_EXACT: rejected only if they are the WHOLE utterance
#                         (people do say "thank you" in a real call).
#   _HALLUCINATION_SUBSTR: never real speech in a call, rejected anywhere.
_HALLUCINATION_EXACT = {
    "շնորհակալություն դիտելու համար",
    "շնորհակալություն ուշադրության համար",
    "բաժանորդագրվեք",
    "բաժանորդագրվեք մեր ալիքին",
    "բաժանորդագրվեք ալիքին",
    "հաջորդ անգամ",
    "շարունակելի",
    "thank you for watching",
    "thanks for watching",
    "спасибо за просмотр",
    "продолжение следует",
    "you",
}
_HALLUCINATION_SUBSTR = (
    "amara.org",
    "subtitles by",
    "subtitled by",
    "ենթագրեր",  # "subtitles"
    "ենթագրերը",
    "субтитр",
    "редактор субтитров",
    "www.",
    "http",
)

_ARMENIAN_RE = re.compile(r"[Ա-֏ﬓ-ﬗ]")
_WS_RE = re.compile(r"\s+")
_CHAR_LOOP_RE = re.compile(r"(\w)\1{4,}")  # same letter 5+ times: "եկկկկկկ" (never real Armenian)
_PUNCT_STRIP = " \t\n.,!?;:\"'«»()[]…—-։՝՜՞"


def normalize(text: str) -> str:
    """Lowercase, NFC, collapse spaces and strip edge punctuation (for matching only)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _WS_RE.sub(" ", text)
    return text.strip(_PUNCT_STRIP)


def collapse_repeats(text: str, max_n: int = 6, keep: int = 1) -> str:
    """Collapse runs of a repeated word n-gram (n = 1..max_n) down to `keep` copies.

    Whisper's failure mode on noisy audio is a loop like
    "ու ու ու ու ու" or "Ես չգիտեմ, Ես չգիտեմ, Ես չգիտեմ, ...".
    A run of 3+ copies is collapsed; 2 copies are left alone ("շատ շատ" is fine).
    """
    words = text.split()
    if len(words) < 3:
        return text

    def key(w: str) -> str:
        return w.lower().strip(_PUNCT_STRIP)

    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            out: list[str] = []
            while i < len(words):
                gram = [key(w) for w in words[i : i + n]]
                if len(gram) < n:
                    out.extend(words[i:])
                    break
                reps = 1
                while [key(w) for w in words[i + reps * n : i + (reps + 1) * n]] == gram:
                    reps += 1
                if reps >= 3:
                    out.extend(words[i : i + keep * n])
                    i += reps * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return " ".join(words)


def armenian_ratio(text: str) -> float:
    """Share of letters in `text` that are Armenian script (1.0 if no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(bool(_ARMENIAN_RE.match(c)) for c in letters) / len(letters)


def compression_ratio(text: str) -> float:
    """Same measure Whisper uses: len(utf8) / len(zlib(utf8)). High = repetitive."""
    import zlib

    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def check_hallucination(
    text: str,
    audio_s: float,
    avg_logprob: float,
    no_speech_prob: float,
    cfg: WhisperConfig,
) -> str:
    """Return a rejection reason, or "" if the text looks like real speech."""
    norm = normalize(text)
    if not norm:
        return "empty"
    if avg_logprob < cfg.min_avg_logprob:
        return f"low confidence (avg_logprob {avg_logprob:.2f})"
    if no_speech_prob > cfg.max_no_speech_prob and avg_logprob < cfg.no_speech_logprob:
        return f"no speech (p={no_speech_prob:.2f}, avg_logprob {avg_logprob:.2f})"
    if norm in _HALLUCINATION_EXACT:
        return f"known hallucination phrase: {norm!r}"
    for s in _HALLUCINATION_SUBSTR:
        if s in norm:
            return f"known hallucination phrase: contains {s!r}"
    m = _CHAR_LOOP_RE.search(norm)
    if m:
        return f"character loop ({m.group(0)[:8]!r}...)"
    ratio = armenian_ratio(norm)
    if ratio < cfg.min_armenian_ratio:
        return f"not Armenian script ({ratio:.0%} Armenian letters)"
    cr = compression_ratio(norm)
    if len(norm) > 40 and cr > cfg.max_compression_ratio:
        return f"repetitive (compression ratio {cr:.2f})"
    cps = len(norm) / max(audio_s, 0.5)
    if cps > cfg.max_chars_per_s:
        return f"too much text for the audio ({cps:.0f} chars/s)"
    return ""


def normalize_level(
    pcm: np.ndarray, target_peak: float = 0.9, max_gain: float = 10.0
) -> tuple[np.ndarray, float]:
    """Scale audio so its (click-robust) peak reaches target_peak; returns (audio, gain)."""
    if not len(pcm):
        return pcm, 1.0
    peak = float(np.percentile(np.abs(pcm), 99.9))
    if peak <= 0:
        return pcm, 1.0
    gain = min(max_gain, target_peak / peak)
    if gain <= 1.0:
        return pcm, 1.0  # never attenuate
    return np.clip(pcm * gain, -1.0, 1.0).astype(np.float32), gain


def clean_text(text: str) -> str:
    """Tidy decoder output for display/translation (keeps punctuation)."""
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text).strip()
    return collapse_repeats(text)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class WhisperSTT:
    """faster-whisper wrapper: load once, warm up, then `transcribe()` per segment.

    Thread-safe: calls are serialized with a lock, since the pipeline runs a
    single GPU worker anyway (PLAN.md §2) and concurrent decodes on one GPU
    only make each one slower.
    """

    def __init__(self, cfg: WhisperConfig | None = None) -> None:
        self.cfg = cfg or WhisperConfig()
        self._lock = threading.Lock()
        self._model = self._load()
        if self.cfg.warmup:
            self.warmup()

    # ---------------------------------------------------------------- setup
    def _load(self):
        # Windows: importing torch first makes the cuDNN/cuBLAS DLLs bundled
        # with torch visible to CTranslate2 ("cudnn_ops64_9.dll not found" fix).
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        import ctranslate2
        from faster_whisper import WhisperModel

        c = self.cfg
        if os.path.isdir(c.model):
            _check_local_model(c.model)
        if c.device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            raise RuntimeError(
                "device='cuda' but CTranslate2 sees no CUDA GPU. Check the NVIDIA driver, "
                "or run with device='cpu' (much slower)."
            )
        if c.device == "cuda":
            supported = ctranslate2.get_supported_compute_types("cuda", c.device_index)
            if c.compute_type not in supported and c.compute_type not in ("auto", "default"):
                log.warning("compute_type %r not supported on this GPU (supported: %s)",
                            c.compute_type, sorted(supported))
        t0 = time.perf_counter()
        model = WhisperModel(
            c.model,
            device=c.device,
            device_index=c.device_index,
            compute_type=c.compute_type,
            download_root=c.download_root,
            local_files_only=c.local_files_only,
        )
        log.info(
            "Loaded Whisper %r on %s (%s) in %.1f s",
            c.model, c.device, c.compute_type, time.perf_counter() - t0,
        )
        if not model.model.is_multilingual:
            raise ValueError(f"{c.model!r} is English-only; an Armenian-capable model is needed")
        return model

    def warmup(self) -> None:
        """One dummy decode so CUDA kernels/allocations happen before the first real phrase."""
        rng = np.random.default_rng(0)
        noise = (0.01 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)
        t0 = time.perf_counter()
        self._decode(noise)
        log.info("Whisper warm-up done in %.2f s", time.perf_counter() - t0)

    # ------------------------------------------------------------------ API
    def transcribe_segment(self, segment: Segment) -> SttResult:
        res = self.transcribe(segment.pcm)
        res.segment_id = segment.id
        return res

    def transcribe(self, pcm: np.ndarray) -> SttResult:
        """Transcribe 16 kHz mono float32 audio and filter hallucinations."""
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        audio_s = len(pcm) / SAMPLE_RATE
        if audio_s < self.cfg.min_audio_s:
            return SttResult(text="", ok=False, reason="audio too short", audio_s=audio_s)

        peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
        if peak < 1e-4:
            return SttResult(text="", ok=False, reason="digital silence", audio_s=audio_s)

        gain = 1.0
        if self.cfg.normalize_gain:
            pcm, gain = normalize_level(pcm, self.cfg.target_peak, self.cfg.max_gain)

        t0 = time.perf_counter()
        parts, info = self._decode(pcm)
        elapsed = time.perf_counter() - t0

        raw = "".join(p.text for p in parts).strip()
        avg_lp, no_speech = _aggregate(parts)
        text = clean_text(raw)
        reason = check_hallucination(text, audio_s, avg_lp, no_speech, self.cfg)
        n_raw, n_clean = len(raw.split()), len(text.split())
        if not reason and n_raw >= 6 and n_clean <= n_raw / 3:
            # The utterance was mostly one phrase looping: a decoder failure, not speech.
            reason = f"repetition loop ({n_raw} -> {n_clean} words after collapsing)"

        return SttResult(
            text="" if reason else text,
            ok=not reason,
            reason=reason,
            raw_text=raw,
            avg_logprob=avg_lp,
            no_speech_prob=no_speech,
            compression_ratio=compression_ratio(text),
            audio_s=audio_s,
            elapsed_s=elapsed,
            extra={
                "temperatures": [p.temperature for p in parts],
                "n_parts": len(parts),
                "gain": round(gain, 2),
            },
        )

    # ------------------------------------------------------------ internals
    def _decode(self, pcm: np.ndarray):
        c = self.cfg
        with self._lock:
            segments, info = self._model.transcribe(
                pcm,
                language=c.language,
                task="transcribe",
                beam_size=c.beam_size,
                temperature=list(c.temperatures),
                max_new_tokens=min(440, int(20 + c.max_tokens_per_s * len(pcm) / SAMPLE_RATE)),
                compression_ratio_threshold=c.compression_ratio_threshold,
                log_prob_threshold=c.log_prob_threshold,
                no_speech_threshold=c.no_speech_threshold,
                condition_on_previous_text=False,
                initial_prompt=c.initial_prompt,
                hotwords=c.hotwords,
                no_repeat_ngram_size=c.no_repeat_ngram_size,
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=False,
            )
            # `segments` is a lazy generator: decoding happens while we iterate.
            return list(segments), info


def _check_local_model(path: str) -> None:
    """Fail early, with the fix, for converted folders faster-whisper would misread."""
    from stt.convert_model import check_tokenizer_layout

    if not os.path.exists(os.path.join(path, "preprocessor_config.json")):
        raise ValueError(
            f"{path}: preprocessor_config.json missing, so faster-whisper would use 80 mel "
            f"bins instead of 128. Fix: python -m stt.convert_model --fix {path}"
        )
    problem = check_tokenizer_layout(path)
    if problem:
        raise ValueError(f"{path}: not usable with faster-whisper: {problem}")


def _aggregate(parts: Iterable) -> tuple[float, float]:
    """Duration-weighted avg_logprob and max no_speech_prob over decoded sub-segments."""
    parts = list(parts)
    if not parts:
        return -10.0, 1.0
    weights = [max(p.end - p.start, 0.1) for p in parts]
    avg_lp = float(np.average([p.avg_logprob for p in parts], weights=weights))
    no_speech = float(max(p.no_speech_prob for p in parts))
    return avg_lp, no_speech


# --------------------------------------------------------------------------- #
# Standalone check:  python -m stt.whisper <file.wav> [--segment]
# --------------------------------------------------------------------------- #
def offline_segments(audio: np.ndarray) -> list[Segment]:
    """Cut a whole 16 kHz file into phrases with the live Silero segmenter (no pacing)."""
    from audio.segmenter import FRAME_SAMPLES, Segmenter, SileroVAD
    from core.types import AudioChunk

    seg = Segmenter("system", SileroVAD())
    out: list[Segment] = []
    audio = np.concatenate([audio, np.zeros(SAMPLE_RATE, dtype=np.float32)])
    for i in range(0, len(audio), FRAME_SAMPLES):
        out += seg.process(AudioChunk("system", i / SAMPLE_RATE, audio[i : i + FRAME_SAMPLES]))
    return out + seg.flush()


def _main() -> None:
    import argparse
    import os
    import sys

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless on Windows

    from audio.file_source import load_audio

    ap = argparse.ArgumentParser(description="Transcribe an Armenian audio file with Whisper.")
    ap.add_argument("file")
    ap.add_argument("--model", default=WhisperConfig.model)
    ap.add_argument("--device", default=WhisperConfig.device)
    ap.add_argument("--compute-type", default=WhisperConfig.compute_type)
    ap.add_argument("--beam-size", type=int, default=WhisperConfig.beam_size)
    ap.add_argument(
        "--segment", action="store_true",
        help="cut the file into phrases with the Silero segmenter first (like live mode)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    print(f"Loading Whisper {args.model!r} on {args.device} ({args.compute_type}) - "
          "the first run downloads the model (~3 GB for large-v3, no progress bar)...",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    stt = WhisperSTT(WhisperConfig(
        model=args.model, device=args.device,
        compute_type=args.compute_type, beam_size=args.beam_size,
    ))
    print(f"Model ready in {time.perf_counter() - t0:.1f} s. Transcribing {args.file} ...",
          file=sys.stderr, flush=True)
    audio = load_audio(args.file)

    if args.segment:
        chunks = [(s.t_start, s.t_end, s.pcm) for s in offline_segments(audio)]
    else:
        chunks = [(0.0, len(audio) / SAMPLE_RATE, audio)]

    for t_start, t_end, pcm in chunks:
        r = stt.transcribe(pcm)
        temps = r.extra.get("temperatures", [])
        fb = " T>0!" if any(t > 0 for t in temps) else ""  # temperature fallback = re-decode
        head = f"[{t_start:6.2f}-{t_end:6.2f}s] {r.elapsed_s:4.2f}s  lp={r.avg_logprob:5.2f}{fb}"
        if r.ok:
            print(f"{head}  HY: {r.text}")
        else:
            print(f"{head}  -- dropped: {r.reason}  (raw: {r.raw_text!r})")


if __name__ == "__main__":
    _main()
