"""Armenian -> English translation with NLLB-200 (CTranslate2).

Input is the Armenian text from the STT stage (`SttResult.text`, one phrase of
<= ~8 s of speech). Output is an `MtResult`: the English text, a confidence
number, and the time the call took (for the 5 s latency budget).

    mt = NllbTranslator(NllbConfig(model="models/nllb-200-distilled-1.3B"))  # loads + warms up
    res = mt.translate("Բարև, ինչպե՞ս ես։")
    if res.ok:
        print(res.text)            # "Hi, how are you?"

The model folder is made once with:
    python -m translate.convert_model facebook/nllb-200-distilled-1.3B models/nllb-200-distilled-1.3B

Which model is used is only a config value, so the MT benchmark can swap the
DECISION.md candidates (NLLB 600M / 1.3B / 3.3B) without code changes.

Design choices (see PLAN.md §3-§5, DECISION.md stage 4):
  * NLLB is a sentence-level model. A phrase that contains several sentences
    is split on sentence-final punctuation (Armenian "։", ".", "!", "?", "…")
    and the pieces are translated as ONE batch, which is about as fast as one
    sentence on the GPU and avoids the model dropping a second sentence.
  * Output length is capped relative to the input (`max_len_ratio`), so a
    looping decoder can't run to 256 tokens and blow the latency budget.
  * Post-filter: collapse repetition loops ("I'm, I'm, I'm, ...") and flag,
    but still show, low-confidence or untranslated output. STT already threw
    away noise/hallucinations; if a phrase got this far, a rough English line
    is more useful in a live call than nothing, so we mark it instead.
  * Calls are serialized with a lock: the pipeline runs one GPU worker for
    STT + MT (PLAN.md §2), and Whisper is also CTranslate2 on the same GPU.

Standalone checks (no live audio needed):
    python -m translate.nllb "Բարև, ինչպե՞ս ես։"
    python -m translate.nllb --file phrases_hy.txt
    python -m translate.nllb                              # type/paste lines, Ctrl+Z/Ctrl+D ends
    python -m translate.nllb --wav recordings/mic.wav     # Whisper -> NLLB on a recording
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# PLAN.md §4: translation should take <= 0.5 s per phrase.
MT_BUDGET_S = 0.5


# --------------------------------------------------------------------------- #
# Config and result types
# --------------------------------------------------------------------------- #
@dataclass
class NllbConfig:
    # Path to a CTranslate2 model folder (made by translate.convert_model).
    model: str = "models/nllb-200-distilled-1.3B"
    # Where to get the tokenizer if the model folder has none (needs network once).
    tokenizer: str | None = None
    device: str = "cuda"  # "cuda" | "cpu" | "auto"
    device_index: int = 0
    # Same reasoning as WhisperConfig: int8 weights halve VRAM (1.3B: ~1.4 GB
    # instead of ~2.7 GB) so NLLB fits next to Whisper on a small card; GTX 16xx
    # have no tensor cores, so float16 brings no speed-up there. On an RTX card
    # with VRAM to spare, try "float16" / "int8_float16".
    compute_type: str = "int8_float32"
    cpu_threads: int = 0  # 0 = CTranslate2 default (only used on CPU)

    src_lang: str = "hye_Armn"
    tgt_lang: str = "eng_Latn"

    # Decoding
    beam_size: int = 4  # 1 = greedy (faster), 4 = usual NLLB setting; benchmark it
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0  # >1 discourages loops but can hurt real repeats
    no_repeat_ngram_size: int = 0
    max_input_tokens: int = 256  # an 8 s phrase is ~40-80 tokens
    # Output cap: max_len_ratio * input tokens + max_len_extra. English needs
    # FEWER tokens than Armenian in NLLB's vocabulary, so 2x is very generous.
    max_len_ratio: float = 2.0
    max_len_extra: int = 10

    split_sentences: bool = True
    min_sentence_words: int = 2  # shorter pieces are glued to their neighbour

    # Post-filter (flag only, the text is still returned)
    low_confidence_logprob: float = -1.5  # avg token log-prob below this -> low_confidence
    max_armenian_ratio: float = 0.3  # more Armenian letters than this in the EN -> untranslated

    warmup: bool = True


@dataclass
class MtResult:
    """What the translation stage hands to the UI (or to the debug log)."""

    text: str  # English text ("" if nothing to translate)
    ok: bool  # False -> nothing usable (empty input or empty output)
    source: str = ""  # the Armenian text that was actually translated (after cleanup)
    low_confidence: bool = False  # True -> the UI can mark the line, e.g. with "(?)"
    reason: str = ""  # why it's not ok / why it's low-confidence
    raw_text: str = ""  # decoder output before cleanup
    avg_logprob: float = 0.0  # token-weighted over sentences
    n_sentences: int = 0
    elapsed_s: float = 0.0  # wall time spent in the model
    segment_id: int | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Text prep and post-filter (pure functions, unit-testable)
# --------------------------------------------------------------------------- #
_ARMENIAN_RE = re.compile(r"[Ա-֏ﬓ-ﬗ]")
_WS_RE = re.compile(r"\s+")
# Whisper sometimes writes the Armenian full stop "։" (U+0589) as an ASCII colon
# or as "․" (U+2024). NLLB's Armenian training data uses "։", so map them back
# when they follow an Armenian letter and end a sentence.
_ASCII_FULL_STOP_RE = re.compile(r"(?<=[Ա-֏])[:․](?=\s|$)")
# Split AFTER sentence-final punctuation followed by whitespace. The Armenian
# question / exclamation marks "՞" "՜" sit INSIDE the word, so they are not
# sentence boundaries here; the following "։" or "?" is.
_SENT_SPLIT_RE = re.compile(r"(?<=[։.!?…])\s+")
_EDGE_PUNCT = " \t\n.,!?;:\"'«»()[]…—-։՝՜՞"


def prepare_source(text: str) -> str:
    """Normalize STT output for NLLB: NFC, single spaces, Armenian full stops."""
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text).strip()
    return _ASCII_FULL_STOP_RE.sub("։", text)


def split_sentences(text: str, min_words: int = 2) -> list[str]:
    """Split a phrase into sentences; very short pieces are merged with a neighbour.

    "Այո։ Վաղը կգամ։"  ->  ["Այո։ Վաղը կգամ։"]          ("Այո։" alone is 1 word)
    "Բարև։ Ինչպե՞ս ես։ Ես լավ եմ։" -> ["Բարև։ Ինչպե՞ս ես։", "Ես լավ եմ։"]
    """
    parts = [p for p in _SENT_SPLIT_RE.split(text.strip()) if p]
    out: list[str] = []
    pending = ""  # short pieces waiting to be glued onto the NEXT sentence
    for p in parts:
        p = (pending + " " + p).strip() if pending else p
        pending = ""
        if len(p.split()) < min_words:
            pending = p
        else:
            out.append(p)
    if pending:  # short tail: glue onto the previous sentence
        if out:
            out[-1] = out[-1] + " " + pending
        else:
            out.append(pending)
    return out


def collapse_repeats(text: str, max_n: int = 6) -> str:
    """Collapse 3+ back-to-back copies of a word n-gram (n = 1..max_n) to one copy.

    NLLB's failure mode on garbled input is a loop such as
    "I don't know, I don't know, I don't know, ...". Two copies are left alone
    ("very, very good" is real English).
    """
    words = text.split()
    if len(words) < 3:
        return text

    def key(w: str) -> str:
        return w.lower().strip(_EDGE_PUNCT)

    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            out: list[str] = []
            i = 0
            while i < len(words):
                gram = [key(w) for w in words[i : i + n]]
                if len(gram) < n:
                    out.extend(words[i:])
                    break
                reps = 1
                while [key(w) for w in words[i + reps * n : i + (reps + 1) * n]] == gram:
                    reps += 1
                if reps >= 3:
                    out.extend(words[i : i + n])
                    i += reps * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return " ".join(words)


def armenian_ratio(text: str) -> float:
    """Share of letters in `text` that are Armenian script (0.0 if no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(_ARMENIAN_RE.match(c)) for c in letters) / len(letters)


def clean_output(text: str) -> str:
    """Tidy decoder output for display."""
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text).strip()
    text = collapse_repeats(text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def check_output(text: str, avg_logprob: float, cfg: NllbConfig) -> str:
    """Return a low-confidence reason, or "" if the translation looks normal."""
    if avg_logprob < cfg.low_confidence_logprob:
        return f"low confidence (avg_logprob {avg_logprob:.2f})"
    ratio = armenian_ratio(text)
    if ratio > cfg.max_armenian_ratio:
        return f"partly untranslated ({ratio:.0%} Armenian letters)"
    return ""


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class NllbTranslator:
    """CTranslate2 NLLB wrapper: load once, warm up, then `translate()` per phrase.

    `translator` / `tokenizer` can be passed in to test the logic without the
    real model (see tests/test_translate.py).
    """

    def __init__(self, cfg: NllbConfig | None = None, *, translator=None, tokenizer=None) -> None:
        self.cfg = cfg or NllbConfig()
        self._lock = threading.Lock()
        if translator is None or tokenizer is None:
            translator, tokenizer = self._load()
        self._translator = translator
        self._tokenizer = tokenizer
        if self.cfg.warmup:
            self.warmup()

    # ---------------------------------------------------------------- setup
    def _load(self):
        # Windows: importing torch first makes the cuDNN/cuBLAS DLLs bundled
        # with torch visible to CTranslate2 (same fix as in stt/whisper.py).
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        import ctranslate2
        from transformers import AutoTokenizer

        from translate.convert_model import check_model_dir

        c = self.cfg
        if not os.path.isdir(c.model):
            raise FileNotFoundError(
                f"NLLB model folder {c.model!r} not found. Create it once with:\n"
                f"  python -m translate.convert_model facebook/nllb-200-distilled-1.3B {c.model}"
            )
        problem = check_model_dir(c.model, c.src_lang, c.tgt_lang, need_tokenizer=c.tokenizer is None)
        if problem:
            raise ValueError(f"{c.model}: {problem}")

        if c.device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
            raise RuntimeError(
                "device='cuda' but CTranslate2 sees no CUDA GPU. Check the NVIDIA driver, "
                "or run with device='cpu' (slower)."
            )
        if c.device == "cuda":
            supported = ctranslate2.get_supported_compute_types("cuda", c.device_index)
            if c.compute_type not in supported and c.compute_type not in ("auto", "default"):
                log.warning("compute_type %r not supported on this GPU (supported: %s)",
                            c.compute_type, sorted(supported))

        t0 = time.perf_counter()
        translator = ctranslate2.Translator(
            c.model,
            device=c.device,
            device_index=c.device_index,
            compute_type=c.compute_type,
            intra_threads=c.cpu_threads,
        )
        tokenizer = _load_tokenizer(AutoTokenizer, c.tokenizer or c.model, c.src_lang)
        log.info("Loaded NLLB %r on %s (%s) in %.1f s",
                 c.model, c.device, c.compute_type, time.perf_counter() - t0)
        return translator, tokenizer

    def warmup(self) -> None:
        """One dummy translation so CUDA kernels/allocations happen before the first real phrase."""
        t0 = time.perf_counter()
        self._translate_sentences(["Բարև ձեզ, ինչպե՞ս եք։"])
        log.info("NLLB warm-up done in %.2f s", time.perf_counter() - t0)

    # ------------------------------------------------------------------ API
    def translate(self, text: str, segment_id: int | None = None) -> MtResult:
        """Translate one Armenian phrase to English."""
        source = prepare_source(text)
        if not source.strip(_EDGE_PUNCT):
            return MtResult(text="", ok=False, reason="empty input", segment_id=segment_id)

        sentences = (split_sentences(source, self.cfg.min_sentence_words)
                     if self.cfg.split_sentences else [source])

        t0 = time.perf_counter()
        outs = self._translate_sentences(sentences)
        elapsed = time.perf_counter() - t0

        raw = " ".join(o for o, _, _ in outs).strip()
        n_tok = sum(n for _, _, n in outs)
        avg_lp = (sum(lp * n for _, lp, n in outs) / n_tok) if n_tok else -10.0
        en = clean_output(raw)
        if not en.strip(_EDGE_PUNCT):
            return MtResult(text="", ok=False, source=source, reason="empty output",
                            raw_text=raw, avg_logprob=avg_lp, n_sentences=len(sentences),
                            elapsed_s=elapsed, segment_id=segment_id)
        flag = check_output(en, avg_lp, self.cfg)
        return MtResult(
            text=en,
            ok=True,
            source=source,
            low_confidence=bool(flag),
            reason=flag,
            raw_text=raw,
            avg_logprob=avg_lp,
            n_sentences=len(sentences),
            elapsed_s=elapsed,
            segment_id=segment_id,
            extra={"sentences": sentences} if len(sentences) > 1 else {},
        )

    def translate_stt(self, stt_result) -> MtResult:
        """Convenience for the pipeline: translate an `stt.whisper.SttResult`."""
        return self.translate(stt_result.text, segment_id=getattr(stt_result, "segment_id", None))

    # ------------------------------------------------------------ internals
    def _translate_sentences(self, sentences: list[str]) -> list[tuple[str, float, int]]:
        """Translate a batch of sentences. Returns (english, avg_logprob, n_tokens) per sentence."""
        c = self.cfg
        tok = self._tokenizer
        sources = []
        for s in sentences:
            ids = tok(s, truncation=True, max_length=c.max_input_tokens)["input_ids"]
            sources.append(tok.convert_ids_to_tokens(ids))
        longest = max(len(s) for s in sources)
        max_len = min(256, int(c.max_len_extra + c.max_len_ratio * longest))

        with self._lock:
            results = self._translator.translate_batch(
                sources,
                target_prefix=[[c.tgt_lang]] * len(sources),
                beam_size=c.beam_size,
                max_input_length=c.max_input_tokens,
                max_decoding_length=max_len,
                length_penalty=c.length_penalty,
                repetition_penalty=c.repetition_penalty,
                no_repeat_ngram_size=c.no_repeat_ngram_size,
                # Translator scores are log-prob / length**length_penalty, so with
                # length_penalty=1.0 the score is the average log-prob per token.
                # (Translator has no `normalize_scores` arg; that's Generator-only.)
                return_scores=True,
            )

        out = []
        for r in results:
            tokens = r.hypotheses[0]
            if tokens and tokens[0] == c.tgt_lang:  # drop the forced language token
                tokens = tokens[1:]
            text = tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)
            score = float(r.scores[0]) if r.scores else 0.0
            out.append((text, score, max(len(tokens), 1)))
        return out


def _load_tokenizer(auto_tokenizer, path: str, src_lang: str):
    """Load the NLLB tokenizer without transformers' false "incorrect regex pattern" warning.

    Recent transformers versions print a Mistral-specific warning ("set
    fix_mistral_regex=True") for any tokenizer.json they don't recognise.
    NLLB is a SentencePiece tokenizer and is not affected, so we silence the
    transformers logger for this one call only and check the result instead.
    """
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        hf_logging = None
    old = hf_logging.get_verbosity() if hf_logging else None
    if hf_logging:
        hf_logging.set_verbosity_error()
    try:
        tok = auto_tokenizer.from_pretrained(path, src_lang=src_lang)
    finally:
        if hf_logging:
            hf_logging.set_verbosity(old)
    ids = tok("Բարև ձեզ")["input_ids"]
    first = tok.convert_ids_to_tokens(ids[:1])[0]
    if first != src_lang:
        raise ValueError(f"tokenizer at {path!r} does not start input with {src_lang!r} "
                         f"(got {first!r}); re-run translate.convert_model")
    return tok


# --------------------------------------------------------------------------- #
# Standalone check:  python -m translate.nllb ["text" ... | --file F | --wav F]
# --------------------------------------------------------------------------- #
def _print_result(hy: str, r: MtResult, prefix: str = "") -> None:
    over = "  OVER BUDGET" if r.elapsed_s > MT_BUDGET_S else ""
    head = f"{prefix}{r.elapsed_s:4.2f}s lp={r.avg_logprob:5.2f}{over}"
    print(f"{head}\n   HY: {hy}")
    if r.ok:
        mark = f"   (?) {r.reason}" if r.low_confidence else ""
        print(f"   EN: {r.text}{mark}")
    else:
        print(f"   -- no translation: {r.reason}")


def _main() -> None:
    import argparse
    import sys

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    for stream in (sys.stdout, sys.stdin):  # Windows consoles default to cp1252/cp437
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Translate Armenian text to English with NLLB-200.")
    ap.add_argument("text", nargs="*", help="Armenian phrase(s); none = read lines from stdin")
    ap.add_argument("--file", help="UTF-8 text file, one Armenian phrase per line")
    ap.add_argument("--wav", help="audio file: run Whisper STT (segmented like live mode) then NLLB")
    ap.add_argument("--model", default=NllbConfig.model)
    ap.add_argument("--device", default=NllbConfig.device)
    ap.add_argument("--compute-type", default=NllbConfig.compute_type)
    ap.add_argument("--beam-size", type=int, default=NllbConfig.beam_size)
    ap.add_argument("--no-split", action="store_true", help="don't split phrases into sentences")
    ap.add_argument("--stt-model", default=None, help="Whisper model for --wav (default: WhisperConfig)")
    ap.add_argument("--stt-beam-size", type=int, default=None,
                    help="Whisper beam size for --wav (default: WhisperConfig)")
    ap.add_argument("--stt-compute-type", default=None,
                    help="Whisper compute type for --wav (default: WhisperConfig)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    print(f"Loading NLLB {args.model!r} on {args.device} ({args.compute_type}) ...",
          file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    mt = NllbTranslator(NllbConfig(
        model=args.model, device=args.device, compute_type=args.compute_type,
        beam_size=args.beam_size, split_sentences=not args.no_split,
    ))
    print(f"NLLB ready in {time.perf_counter() - t0:.1f} s.", file=sys.stderr, flush=True)

    if args.wav:
        _run_wav(mt, args)
        return

    if args.text:
        lines = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        print("Type or paste Armenian lines (Ctrl+Z then Enter on Windows / Ctrl+D to finish):",
              file=sys.stderr, flush=True)
        lines = (ln.strip() for ln in sys.stdin)

    times = []
    for line in lines:
        if not line:
            continue
        r = mt.translate(line)
        times.append(r.elapsed_s)
        _print_result(line, r)
    _print_summary(times)


def _run_wav(mt: NllbTranslator, args) -> None:
    """Whisper -> NLLB on a recording: shows HY + EN and STT / MT / total time per phrase."""
    import sys

    from audio.file_source import load_audio
    from stt.whisper import WhisperConfig, WhisperSTT, offline_segments

    wcfg = WhisperConfig(device=args.device)
    if args.stt_model:
        wcfg.model = args.stt_model
    if args.stt_beam_size is not None:
        wcfg.beam_size = args.stt_beam_size
    if args.stt_compute_type:
        wcfg.compute_type = args.stt_compute_type
    print(f"Loading Whisper {wcfg.model!r} ({wcfg.compute_type}, beam {wcfg.beam_size}) ...",
          file=sys.stderr, flush=True)
    stt = WhisperSTT(wcfg)
    audio = load_audio(args.wav)
    times = []
    for seg in offline_segments(audio):
        s = stt.transcribe_segment(seg)
        if not s.ok:
            print(f"[{seg.t_start:6.2f}-{seg.t_end:6.2f}s] STT dropped: {s.reason}")
            continue
        r = mt.translate_stt(s)
        times.append(r.elapsed_s)
        _print_result(s.text, r, prefix=f"[{seg.t_start:6.2f}-{seg.t_end:6.2f}s] "
                                        f"STT {s.elapsed_s:4.2f}s + MT ")
    _print_summary(times)


def _print_summary(times: list[float]) -> None:
    if not times:
        return
    ts = sorted(times)
    p95 = ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))]
    print(f"\n{len(ts)} phrase(s): mean {sum(ts) / len(ts):.2f} s, p95 {p95:.2f} s, "
          f"max {ts[-1]:.2f} s (budget {MT_BUDGET_S} s)")


if __name__ == "__main__":
    _main()
