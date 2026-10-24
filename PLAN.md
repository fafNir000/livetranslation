# PLAN — Live Armenian Transcription → English Translation

Status: v2 (2026-10-23), updated to match the implemented code. Decisions are marked ✅. Final choices and the reasons for them are in DECISION.md and README.md.

## 1. Goal

A Python terminal program that captures **microphone + system audio** at the same time. It transcribes spoken **Armenian**, translates it to **English** and prints each segment with **time, speaker, HY text and EN text**. Each line must appear **≤ 5 s** after the phrase ends. Everything runs locally.

Target machine: ✅ Windows laptop, NVIDIA GTX 1650 with **4 GB dedicated VRAM** (CUDA). Windows shows "11.9 GB", but that adds shared system RAM. A model that spills into shared RAM runs ~10× slower, so every model must fit in 4 GB (int8).

Out of scope (per brief): GUI, installer, echo cancellation, named or persistent speakers, saving transcripts, non-Armenian input.

## 2. Architecture at a glance

```
 Mic (WASAPI) ──► Capture ─┐
                           ├─► VAD / Segmenter ─► STT ─► Translation ─► Terminal (print)
 Loopback (WASAPI) ► Capture ┘          │       ║ (parallel)                ▲
                                        └──► Speaker ID (system audio only) ┘
```

Capture (PortAudio callbacks), the segmenter and the pipeline worker each run in their own thread, joined by `queue.Queue`s. STT and translation share **one** GPU worker; speaker ID runs on its own single thread in parallel with STT. Models are loaded once at startup and warmed up with a dummy inference so the first real segment isn't slow.

## 3. Modules, inputs and outputs

| # | Module | Input | Output | Planned implementation |
|---|---|---|---|---|
| 1 | **Audio capture** (`audio/capture.py`) | Device IDs from CLI (`--list-devices`, `--mic`, `--loopback`) | `AudioChunk(source: "mic"\|"system", t: float, pcm: float32[16 kHz mono])`, ~30 ms frames | ✅ PyAudioWPatch, with WASAPI loopback on the output device (no virtual cable). Resample (soxr) and downmix to 16 kHz mono. Loopback sends no data during silence, so chunks are stamped with session time. |
| 2 | **VAD / segmenter** (`audio/segmenter.py`) | Stream of `AudioChunk` per source | `Segment(id, source, t_start, t_end, pcm)` | ✅ Silero VAD, one state machine per source. Close a segment after 500 ms of silence, or after 0.6 s with no audio arriving (loopback idle). Force-cut at 8 s at the quietest recent frame. Drop segments with < 0.4 s of speech. |
| 3 | **STT** (`stt/`) | `Segment.pcm` | `hy_text: str` (+ avg log-prob for filtering) | ✅ faster-whisper with **Chillarmo/whisper-large-v3-turbo-armenian** converted to CTranslate2 (`models/whisper-turbo-hy`, int8_float32, beam 1, no temperature fallback). Language forced to `hy`. Low-confidence or hallucinated output gets filtered out. |
| 4 | **Translation** (`translate/`) | `hy_text` | `en_text: str` | ✅ NLLB-200 distilled 1.3B (CTranslate2, int8_float32, beam 4). `hye_Armn → eng_Latn`. Multi-sentence phrases are split and translated as one batch. |
| 5 | **Speaker ID** (`speakers/`) | `Segment` (source + pcm) | `label: "Me" \| "Speaker N"` | ✅ The mic is always **Me**. For system audio, ✅ **pyannote** embedding model computes a voice embedding per segment, then online clustering assigns the label: cosine similarity against running speaker centroids, with a new speaker created below a tuned threshold. Model: `pyannote/wespeaker-voxceleb-resnet34-LM`, threshold 0.5. The full pyannote diarization pipeline is batch-oriented, so we use its embeddings live. Model is downloaded once (HF token if asked), then runs offline. |
| 6 | **Terminal output** (in `main.py`) | segment time, label, HY, EN | Printed block: `[hh:mm:ss] Speaker 1` / `HY: …` / `EN: …` | ✅ Plain `print` (no Rich, no colours, no separate `ui/` module: the brief asks for a basic terminal). Optional `--debug` shows per-stage latency. |
| 7 | **Orchestrator** (`main.py`) | CLI args | Wires queues and threads, handles clean shutdown (Ctrl+C), prints a latency summary | ✅ argparse; model settings go into the existing `WhisperConfig` / `NllbConfig` / `SpeakerParams` dataclasses. Replay mode via `--mic-file` / `--system-file`. |

**VRAM budget (rough):** Whisper turbo-armenian int8 ~1 GB, NLLB 1.3B int8 ~1.4 GB, pyannote embedding < 0.2 GB, CUDA context ~0.5–1 GB. That totals ~3–3.5 GB of 4 GB. (The original fp16 plan, ~8 GB, assumed 11.9 GB of real VRAM and would not fit.)

## 4. Latency budget (end of phrase → EN line)

| Stage | Target |
|---|---|
| VAD silence wait | 0.5 s |
| STT (≤ 8 s audio, GPU) | ≤ 1.5 s |
| Translation | ≤ 0.5 s |
| Speaker embedding | ≤ 0.2 s (runs in parallel with STT) |
| Queueing / print | ≤ 0.3 s |
| **Total** | **≈ 3 s, leaving ~2 s headroom** |

If both sources talk at once, segments queue for the single GPU. We process them in order of `t_end` and log any segment that breaks the 5 s limit.

## 5. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Latency > 5 s** | Hard requirement | Force-cut long segments. int8 models, greedy STT, no temperature fallback, output length caps, warm-up. Measure per-stage timings on every segment (`--debug`). Turbo instead of large-v3 for speed. |
| **Armenian STT quality** | Low-resource language, so WER varies widely by model | Benchmark first (see §6). Force `language="hy"`. Filter Whisper hallucinations on silence or noise (e.g. repeated phrases, "subtitles by…"). Document the limits honestly in the README (no WER was measured, see §6). |
| **Translation quality** | Transcription errors carry over into the translation | Pick the translator on real transcripts, not clean text. Keep segments sentence-sized. |
| **WASAPI loopback setup** | Device names, sample rates (often 48 kHz stereo), default device changes mid-call, headphones vs speakers | `--list-devices` helper. Explicit resampling. Test with headphones and speakers. Write a step-by-step README section, and rehearse before the interview. |
| **Speaker ID stability** | Short segments give noisy embeddings, which leads to over-splitting or label swaps | Minimum segment length for clustering. Update centroids with a running mean. Tune the threshold on our clips. Note overlapping speech within one segment as a known limitation. |
| **pyannote access** | Gated HF model, Windows install quirks | Accept the model terms and cache weights once. Pin versions. Never commit weights or tokens (`.gitignore`, `.env`). |
| **GPU contention** | Three models on one GPU | Single GPU worker thread for STT + translation. Measure VRAM at startup. |

## 6. Testing plan

**Test data:** ✅ our own recorded Armenian audio in `recordings/` and short clips for speaker tuning in `clips/` (both git-ignored). Hand-written reference transcripts were planned but not made, so no WER/CER was computed.

| Part | How we test it | Pass criterion |
|---|---|---|
| Audio capture | `--list-devices`. Record 10 s from mic + loopback to WAV and listen back. | Both streams clean, correct rate, no drift or dropouts |
| VAD / segmenter | Run on clips and compare segment boundaries with the reference turns | No words cut mid-phrase, and no segment > 8 s |
| STT | `bench/stt_speed.py`: time per phrase, VRAM and output text per model / compute type / beam, compared by reading the text (no WER, see above) | ≤ 1.5 s per segment, readable Armenian |
| Translation | `python -m translate.nllb` on typed phrases and `--wav` (STT → MT on a recording); unit tests in `tests/test_translate.py` | Understandable English on most segments, ≤ 0.5 s |
| Speaker ID | `bench/cut_clips.py` + `python -m speakers.pyannote clips/*.wav`: similarity matrix and labels per clip | Same person keeps the same label, and speakers are told apart |
| End-to-end | **Replay mode** (`--mic-file` / `--system-file`) feeds WAV files through the full pipeline in real time. Per-segment latency is shown with `--debug` and summarized on exit. | p95 latency < 5 s |
| Live | Real call with 2+ Armenian speakers on Windows, ideally on the same call app we'll use in the interview | Runs for 10+ min with no crashes and stays under 5 s |

Unit tests (pytest, no GPU) cover the segmenter state machine and the translation text handling / engine logic with a fake model. Clustering and resampling have no unit tests yet.

## 7. Milestones (~20 h)

1. **Audio spike (2 h):** list devices, dual WASAPI capture to WAV.
2. **Record test clips + references (2 h).**
3. **Model benchmark (3 h):** STT and translation, then lock in the choices.
4. **Pipeline (5 h):** segmenter → STT → translation → UI, with replay mode.
5. **Speaker ID (3 h):** pyannote embeddings + online clustering, then threshold tuning.
6. **Hardening (3 h):** latency logging, hallucination filter, error handling, live call tests.
7. **README + chats export (2 h).**
