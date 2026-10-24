# DECISION.md — Model & Technology Choices per Stage

Decision log for each pipeline stage: the options considered, the criteria, what was chosen, and how the choice was checked. Started as a plan on 2026-10-22; this version (2026-10-24) records the final decisions as implemented. The README's "Models" section is a summary of this file.

**Status legend:** ✅ Chosen · ⚪ Considered, not chosen · ❌ Tried and rejected

**Global constraints (from brief):** local models only · Python · ≤ 5 s from end of phrase to English line · runs on my own laptop.

**Hardware:** Windows laptop, NVIDIA GTX 1650 with **4 GB dedicated VRAM**. Windows reports "11.9 GB", but that includes shared system RAM, and a model that spills into it runs ~10× slower. The first plan assumed 11.9 GB of real VRAM and fp16 models (~8 GB in total). Finding this out changed several choices below: **int8 instead of fp16**, and **turbo instead of large-v3**.

## Summary

| Stage | Choice |
|---|---|
| Audio capture | PyAudioWPatch, WASAPI mic + loopback |
| Segmentation | Silero VAD, one state machine per source |
| Speech-to-text | Chillarmo/whisper-large-v3-turbo-armenian, faster-whisper, int8, beam 1 |
| Translation | NLLB-200 distilled 1.3B, CTranslate2, int8, beam 4 |
| Speaker ID | pyannote WeSpeaker ResNet34 embeddings + own online clustering |
| Terminal output | plain `print` |
| Concurrency | threads + queues, one GPU worker, speaker ID in parallel |

---

## Stage 1 — Audio capture (mic + system audio)

**Criteria:** captures both sources at once on Windows, the user still hears the call, minimal setup for the interview, the user can pick devices.

| | Option | Pros | Cons |
|---|---|---|---|
| ✅ | **PyAudioWPatch (WASAPI loopback)** | Loopback on any output device, no driver install, familiar PortAudio API | Windows-only fork; devices are often 48 kHz stereo, so audio must be resampled |
| ⚪ | `soundcard` library | Simple API, WASAPI loopback, cross-platform | Less control over buffer sizes; reported glitches when devices switch |
| ⚪ | `sounddevice` | Popular, solid for the mic | Loopback on Windows is limited, so a second library would still be needed |
| ⚪ | VB-Cable virtual device | Same idea as BlackHole on macOS; works with any capture library | Driver install; the user must reroute call audio and set up monitoring to hear it |

**Decision:** ✅ PyAudioWPatch for both streams, one library and one code path (`audio/capture.py`). Each device is downmixed and resampled to 16 kHz mono with soxr. VB-Cable was not needed.

**How it was checked:** `python -m audio.capture --record 10` records both sources to WAV, with a level meter and an overflow counter; the recordings were then played back. This showed a quirk: **WASAPI loopback sends no data at all while nothing is playing**. Because of that, chunks are stamped with session time, and the segmenter closes an open phrase on an idle timeout instead of waiting for silence frames that never arrive.

---

## Stage 2 — Voice activity detection / segmentation

**Criteria:** cuts at natural phrase ends quickly (the latency budget allows ~0.5 s), robust to background noise and call compression, light enough for the CPU.

| | Option | Pros | Cons |
|---|---|---|---|
| ✅ | **Silero VAD** | Accurate, tiny, fast on CPU, widely used with Whisper | Threshold and minimum silence need tuning |
| ⚪ | WebRTC VAD | Extremely fast, no torch dependency | Noticeably worse on noisy or compressed audio, more false triggers |
| ⚪ | faster-whisper built-in VAD filter | No extra code | Works on whole buffers; it doesn't decide *when* to cut a live stream |
| ⚪ | pyannote segmentation model | Also detects speaker changes | Heavier; overkill for VAD alone |
| ⚪ | Energy threshold | Trivial | Breaks with background noise or music |

**Decision:** ✅ Silero VAD on CPU, one instance and state machine per source (`audio/segmenter.py`), so mic and call audio never mix. Parameters:

- speech starts at probability 0.5 and counts as silence below 0.35 (hysteresis)
- 500 ms of silence closes a phrase; 200 ms of padding is kept on each side
- segments with less than 0.4 s of speech are dropped
- a phrase is force-cut at 8 s, at the quietest frame of the last 1.5 s
- an open phrase is closed after 0.6 s with no audio arriving (loopback gaps)

**How it was checked:** unit tests with a fake VAD (`tests/test_segmenter.py`: pause, two phrases, short blip, forced cut, odd chunk sizes, idle timeout, stream gap), plus cutting real recordings with `python -m stt.whisper <wav> --segment`. Segment boundaries were not compared against hand-marked speaker turns.

---

## Stage 3 — Speech-to-text (Armenian)

**Criteria, in order:** Armenian accuracy → speed (≤ ~1.5 s per segment) → behaviour on silence and noise → fits in VRAM next to translation and speaker ID → runs in faster-whisper / CTranslate2.

| | Option | Size | Reported Armenian results* | Pros | Cons |
|---|---|---|---|---|---|
| ✅ | **Chillarmo/whisper-large-v3-turbo-armenian** | 0.8B | WER 15.3 % / CER 2.9 % (Common Voice 20) | Armenian fine-tune; 4-layer turbo decoder, so fast; ~0.8 GB int8; Apache 2.0 | Trained on Common Voice (read speech), so conversational speech may be worse |
| ⚪ | Whisper large-v3 | 1.55B | CER ~13.8 % FLEURS zero-shot (per BuzzASR card) | Well-known baseline; ~1.6 GB int8 | No Armenian tuning; slower (see below); hallucinates on silence |
| ⚪ | Whisper large-v3-turbo (stock) | 0.8B | Usually a bit worse than large-v3 on low-resource languages | Same speed as the chosen model | Same speed with no Armenian tuning, so the fine-tune is strictly better for us |
| ❌ | BuzzASR/armenian | ~1.5B | WER 11.9 % / CER 3.7 % FLEURS | Best published numbers; MIT license | **Doesn't run in faster-whisper** (see below) |
| ⚪ | SeamlessM4T v2 large | 2.3B | Supports Armenian ASR | Could also translate | Heavier; no CTranslate2 runtime; too big next to NLLB on 4 GB |
| ⚪ | Meta Omnilingual ASR / MMS | 0.3B–7B | Supports Armenian; reported weaker by BuzzASR | Fast CTC variants | CTC output usually lacks punctuation and casing, which hurts translation |

\* The authors' own numbers, mostly on read-speech test sets.

**Decision:** ✅ **Chillarmo/whisper-large-v3-turbo-armenian**, converted with `stt/convert_model.py` to `models/whisper-turbo-hy` and run with faster-whisper: **int8_float32**, beam 1, temperature 0.0 only, language forced to `hy`.

**How it was decided** (`bench/stt_speed.py` and `python -m stt.whisper` on our recordings, GTX 1650):

- **BuzzASR ❌:** its custom tokenizer puts `<|endoftext|>` at id 0. faster-whisper assumes the standard Whisper vocabulary and cannot decode it, so BuzzASR would need the much slower HF `transformers` runtime. `stt/convert_model.py` now detects this layout and refuses it.
- **Whisper large-v3:** int8 took ~1.2 s per phrase at beam 1 and ~1.4–1.5 s at beam 5, with nearly the same text. That leaves little headroom under the 1.5 s budget, and the model has no Armenian tuning. It stays available with `--stt-model large-v3`.
- **Turbo-armenian:** Armenian-specific training plus a turbo decoder. It is faster and smaller, which leaves VRAM for NLLB.
- **int8, not fp16:** GTX 16xx cards have no tensor cores, so fp16 gives no speed-up, while int8 halves VRAM.
- **Beam 1:** beam 5 cost ~0.2–0.3 s per phrase for nearly identical text.
- **No temperature fallback:** every fallback is a full extra decode and doubled or tripled latency in the benchmark. A hallucination filter in `stt/whisper.py` catches bad output instead: low confidence, known subtitle phrases, repetition loops, wrong script, too much text for the audio length.
- Also in `stt/whisper.py`: segments are decoded independently (`condition_on_previous_text=False`), output tokens are capped by audio length, and quiet audio is normalized by up to +20 dB.

**Not done:** WER/CER against hand-written reference transcripts. Models were compared on compatibility, speed, published numbers and by reading their output on our recordings. The README lists this as a limitation.

---

## Stage 4 — Translation (Armenian → English)

**Criteria:** adequacy on *real STT output* (with its errors), ≤ ~0.5 s per phrase, VRAM, robustness to missing punctuation.

| | Option | Size | Pros | Cons |
|---|---|---|---|---|
| ✅ | **NLLB-200 distilled 1.3B** (CTranslate2) | 1.3B | Built for low-resource pairs incl. `hye_Armn`; ~1.4 GB int8; fast | Sentence-level only; can be literal; CC-BY-NC-4.0 (fine for this challenge, not for commercial use) |
| ⚪ | NLLB-200 distilled 600M | 0.6B | Very fast, ~0.6 GB int8 | Lower quality. Kept as a fallback if latency is tight |
| ⚪ | NLLB-200 3.3B | 3.3B | Best NLLB quality | ~3.4 GB even in int8; doesn't fit next to STT on 4 GB |
| ⚪ | MADLAD-400 3B-MT | 3B | Apache 2.0; covers Armenian | Too large and slow for this GPU; hy→en quality unproven |
| ⚪ | SeamlessM4T v2 (text→text or speech→text) | 2.3B | Could replace the separate MT step | Heavier; speech-to-English loses the Armenian transcript we must print |
| ⚪ | Whisper `task="translate"` | — | Same model, no extra VRAM | No Armenian text; lower quality; fine-tuned Armenian models lose this ability |
| ⚪ | Small local LLM (Gemma / Qwen 4–8B, int4) | 4–8B | Fluent, context-aware English | Too slow and big for 4 GB; may invent or "improve" content |

**Decision:** ✅ **NLLB-200 distilled 1.3B**, converted with `translate/convert_model.py` to `models/nllb-200-distilled-1.3B`: CTranslate2 **int8_float32**, beam 4. The tokenizer is saved inside the model folder, so translation runs offline.

- int8, not fp16, for the same reasons as STT: fits on 4 GB, and no tensor cores.
- Multi-sentence phrases are split on `։ . ! ? …` and translated as one batch (about as fast as a single sentence, and NLLB stops dropping the second sentence). Very short pieces are merged with a neighbour.
- Output length is capped relative to the input, and repetition loops are collapsed.
- Low-confidence or partly untranslated output is flagged, not dropped. In a live call a rough English line is more useful than nothing.

**How it was checked:** `python -m translate.nllb` on typed phrases and on real STT output (`--wav <recording>`), with timing per phrase against the 0.5 s budget. Unit tests in `tests/test_translate.py` cover the text handling and the engine with a fake model. **Not done:** chrF scores or a systematic manual rating against 600M / 3.3B.

---

## Stage 5 — Speaker identification (system audio side)

**Fixed rule:** ✅ microphone = **Me**, no model needed.

**Criteria for the system side:** the same person keeps the same label all session; works incrementally (live, not on a finished recording); fast per segment; robust on 1–3 s segments of call-compressed audio.

| | Option | Pros | Cons |
|---|---|---|---|
| ✅ | **pyannote embedding model + our own online clustering** | Strong WeSpeaker ResNet34 embeddings (used inside pyannote's pipelines); simple, explainable logic; a few ms per segment | We write and tune the clustering; a segment containing two speakers gets one label |
| ⚪ | pyannote `speaker-diarization-community-1` (full pipeline) | Best diarization quality, overlap-aware | Built for whole files; rerunning on a growing buffer adds latency and labels can change between runs |
| ⚪ | `diart` (streaming diarization on pyannote models) | Built for real time; handles speaker changes within a stream | Extra dependency; pyannote 4.x compatibility unclear; harder to explain and tune |
| ⚪ | SpeechBrain ECAPA-TDNN + own clustering | Well known, no gated download | Same work as the chosen option, slightly older embeddings |
| ⚪ | NVIDIA NeMo TitaNet / Sortformer | Strong models, streaming variants | Heavy NeMo install on Windows; overkill |

**Decision:** ✅ `pyannote/wespeaker-voxceleb-resnet34-LM` embeddings + online centroid clustering (`speakers/pyannote.py`):

- cosine similarity ≥ **0.5** to a known speaker → same speaker; its centroid is updated (weight capped at 30 s so it can still adapt)
- otherwise, a segment ≥ 1.2 s creates a new speaker; a shorter one goes to the closest speaker without updating its centroid
- segments < 0.6 s are not embedded and reuse the last system-side speaker
- at most 8 speakers, as a guard against runaway splitting
- runs on its own thread, in parallel with STT

**How it was checked:** recordings were cut into 3 s clips with `bench/cut_clips.py`. `python -m speakers.pyannote clips/*.wav` prints the similarity matrix and the label each clip gets, which is how the 0.5 threshold was chosen. It can be changed at runtime with `--speaker-threshold`. **Not done:** label purity on a long multi-speaker replay against hand-marked turns, and the `diart` comparison.

---

## Stage 6 — Terminal output

**Criteria:** shows time, speaker, HY and EN for every phrase; readable live; no flicker; easy to demo.

| | Option | Pros | Cons |
|---|---|---|---|
| ✅ | **Plain `print`** | No dependency; matches the brief's example format exactly | No colours per speaker |
| ⚪ | Rich (coloured console) | Coloured speakers, works in Windows Terminal | Extra dependency; nothing the brief requires |
| ⚪ | Textual (full TUI) | Panels, scrolling, status bar | Much more code; brief says keep it basic |

**Decision:** ✅ Plain `print` in `main.py`, with no separate `ui/` module. The brief asks for a basic terminal, and the block format `[hh:mm:ss] Speaker` / `HY:` / `EN:` needs nothing more. A lock keeps blocks from interleaving, and stdout is switched to UTF-8 so Armenian prints correctly in Windows consoles. `--debug` adds a per-phrase timing line, and the program prints a latency summary on exit. (Rich was the initial plan and is still listed in `requirements.txt`, but nothing uses it.)

---

## Stage 7 — Pipeline / concurrency

**Criteria:** two live inputs, one GPU, predictable latency, easy to read and explain.

| | Option | Pros | Cons |
|---|---|---|---|
| ✅ | **Threads + `queue.Queue`** | Simple; audio callbacks and GPU inference release the GIL; easy to explain | GPU work must be ordered carefully |
| ⚪ | asyncio | Clean structure for I/O | Inference is blocking, so threads are needed anyway |
| ⚪ | multiprocessing | True parallelism | Models duplicated in VRAM, complex IPC |

**Decision:** ✅ Threads + queues (`main.py`):

- capture callbacks → `chunk_q` → `SegmenterWorker` → `seg_q` → one `PipelineWorker`
- STT and translation share **one** GPU worker that handles segments in arrival order. Both are CTranslate2 on the same small GPU, and running them concurrently would only slow each other down.
- speaker ID runs on a one-thread executor in parallel with STT
- all models are loaded and warmed up before listening starts
- replay mode (`--mic-file` / `--system-file`, `audio/file_source.py`) feeds WAV files through the same pipeline in real time for repeatable tests
- a failing segment is logged and skipped; Ctrl+C drains the queue and prints mean / p95 / max latency and the number of phrases over 5 s

---

## VRAM budget (final)

| Component | Est. VRAM |
|---|---|
| STT (Whisper turbo-armenian, int8) | ~1 GB |
| Translation (NLLB 1.3B, int8) | ~1.4 GB |
| Speaker embedding (WeSpeaker ResNet34) | < 0.2 GB |
| CUDA context / buffers | ~0.5–1 GB |
| Silero VAD | CPU, 0 GB |
| **Total** | **~3–3.5 GB of 4 GB dedicated** |

The original fp16 plan (~8 GB) would have spilled into shared RAM on this laptop. `python -m bench.stt_speed <wav>` reports the real VRAM in use and warns when it gets close to the limit.

## Sources consulted (2026-10-22)

- [BuzzASR/armenian model card](https://huggingface.co/BuzzASR/armenian)
- [Chillarmo/whisper-large-v3-turbo-armenian model card](https://huggingface.co/Chillarmo/whisper-large-v3-turbo-armenian)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [SeamlessM4T paper](https://arxiv.org/pdf/2308.11596)
- [Omnilingual ASR paper](https://arxiv.org/pdf/2511.09690)
- [MADLAD-400 paper](https://arxiv.org/pdf/2309.04662)

All accuracy numbers quoted above are the authors' own claims, mostly on read-speech test sets. They were not reproduced on our conversational recordings.
