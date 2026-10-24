# Live Armenian → English translation

A Python program for the terminal that listens to a live call. It captures your **microphone** and the **system audio** (the other people on the call) at the same time, transcribes the Armenian speech, translates it into English and prints each phrase labelled by speaker:

```
[00:01:12] Speaker 1
HY: Բարև, ինչպե՞ս ես։
EN: Hi, how are you?
[00:01:15] Me
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

Everything runs **locally** (no cloud APIs). The goal is **≤ 5 s** from the end of a spoken phrase to its English line.

- Mic audio is always labelled **Me**.
- System audio is split into **Speaker 1, Speaker 2, …** by voice. A speaker keeps the same label for the whole session.
- Timestamps are session time: `00:00:00` is when the program starts listening.

---

## Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Audio device setup](#3-audio-device-setup)
4. [Running](#4-running)
5. [Architecture](#5-architecture)
6. [Models: what was chosen and why](#6-models-what-was-chosen-and-why)
7. [Latency](#7-latency)
8. [Testing](#8-testing)
9. [Project structure](#9-project-structure)
10. [Known limitations](#10-known-limitations)
11. [What I would improve with more time](#11-what-i-would-improve-with-more-time)

---

## 1. Requirements

| | |
|---|---|
| OS | Windows 10/11. System audio is captured with WASAPI loopback, which only exists on Windows. Replay mode (`--mic-file` / `--system-file`) runs anywhere. |
| Python | 3.11, 64-bit |
| GPU | NVIDIA with CUDA 12.x. Developed on a laptop **GTX 1650 (4 GB dedicated VRAM)**. Windows reports "11.9 GB" for this card, but that figure adds shared system RAM. Anything that spills into shared RAM runs about 10× slower, so all models are sized to fit in 4 GB. CPU mode (`--device cpu`) works but is too slow for the 5 s target. |
| Disk | ~3 GB for the converted models |
| Network | Only for installation and the first model download. After that the program can run with `--offline`. |

Main libraries: `faster-whisper` / `ctranslate2` (speech-to-text and translation), `transformers` (NLLB tokenizer), `silero-vad`, `pyannote.audio`, `PyAudioWPatch`, `soxr`, `soundfile`, `torch`. The full list is in `requirements.txt`.

## 2. Installation

```bat
git clone https://github.com/fafNir000/livetranslation.git
cd livetranslation

python -m venv .venv
.venv\Scripts\activate

:: 1) CUDA build of PyTorch FIRST (otherwise pip installs the CPU-only build)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

:: 2) everything else
pip install -r requirements.txt

:: 3) check the GPU is visible
python -c "import torch; print(torch.cuda.is_available())"
```

### Get the models (once, needs network)

Model weights are not in the repository. Two helper scripts download them from Hugging Face and convert them to CTranslate2 int8 in `models/`:

```bat
:: Speech-to-text: Armenian fine-tune of Whisper large-v3-turbo  (~0.8 GB)
python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy

:: Translation: NLLB-200 distilled 1.3B  (~1.4 GB)
python -m translate.convert_model
```

Both scripts also copy the tokenizer and preprocessor files into the model folder and check them. That way nothing has to be fetched at runtime, and faster-whisper picks up the right tokenizer and 128 mel bins.

The speaker-embedding model (`pyannote/wespeaker-voxceleb-resnet34-LM`) is downloaded automatically into the Hugging Face cache on the first run. If Hugging Face asks for authentication, accept the model's terms on its page and create a `.env` file in the project root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

`.env`, `models/`, `recordings/` and `clips/` are git-ignored. No weights, audio or keys are committed.

**Run the program once with network access** (e.g. a replay test, see §4) so the pyannote model gets cached. After that, `--offline` works.

## 3. Audio device setup

No virtual cable or driver is needed. PyAudioWPatch exposes every output device as a **loopback** input, so the program records what your speakers or headphones play while you keep hearing the call normally.

1. Start the call app (Zoom / Meet / Teams…) and choose the output device you will listen on.
2. List devices:
   ```bat
   python main.py --list-devices
   ```
   ```
   Microphones (use with --mic):
    * [  1] Microphone Array (Realtek)  (48000 Hz, 2 ch)

   System audio / loopback (use with --loopback):
    * [ 17] Headphones (Realtek) [Loopback]  (48000 Hz, 2 ch)

   * = default device
   ```
3. Pick the microphone you speak into and the **loopback of the device the call plays on**. Without flags, the Windows defaults (`*`) are used.
4. Headphones are recommended. With speakers, the mic also picks up the other side. The phrase is then printed twice: once as the speaker and once as "Me" (echo handling is out of scope).

To check levels before a call, record 10 s from both sources to WAV:

```bat
python -m audio.capture --mic 1 --loopback 17 --record 10 --out recordings
```

## 4. Running

```bat
:: live, default devices
python main.py

:: live, chosen devices, no network
python main.py --mic 1 --loopback 17 --offline

:: replay test: WAV files are fed through the full pipeline in real time
python main.py --system-file recordings\other\system.wav --mic-file recordings\me\mic.wav --debug
```

Model loading and warm-up take a while. Speak once `Listening... press Ctrl+C to stop.` appears. **Ctrl+C** stops capture, finishes the phrases already queued and prints a latency summary.

### Options

| Flag | Meaning |
|---|---|
| `--list-devices` | Print microphones and loopback devices, then exit |
| `--mic N`, `--loopback N` | Device indices (default: Windows defaults) |
| `--mic-file F`, `--system-file F` | **Replay mode**: play WAV files as mic / system audio instead of live capture. Either one alone is fine. |
| `--system-offset S` | Replay mode: start the system file S seconds later |
| `--offline` | Never contact Hugging Face; load models from disk only |
| `--debug` | Show per-stage timings for each phrase, the segments the filter dropped, and info logs |
| `--device cuda\|cpu` | Device for all models (default `cuda`) |
| `--stt-model`, `--stt-compute-type`, `--stt-beam-size` | Override the speech-to-text model (default `models/whisper-turbo-hy`, `int8_float32`, beam 1) |
| `--mt-model`, `--mt-compute-type`, `--mt-beam-size` | Override the translation model (default `models/nllb-200-distilled-1.3B`, `int8_float32`, beam 4) |
| `--speaker-threshold X` | Cosine similarity needed to match a known speaker (default 0.5) |
| `--speaker-device` | Run pyannote on another device, e.g. `cpu` to save VRAM |
| `--no-speaker-id` | Skip pyannote; all system audio is "Speaker 1" (fallback) |

With `--debug`, each phrase gets one more line:

```
[00:00:07] Speaker 1
HY: ...
EN: ...
    [latency 2.31s | queue 0.00s vad 0.52s stt 1.05s mt 0.38s spk 0.04s | audio 3.4s]
```

- `latency`: end of phrase → English line printed (the 5 s requirement)
- `vad`: wait for the pause that closes the phrase
- `queue`: time the phrase waited for the GPU
- `stt`, `mt`, `spk`: model time per stage
- `audio`: length of the phrase

## 5. Architecture

```
 Mic (WASAPI) ──────► AudioCapture ─┐
                                    ├─► chunk_q ─► SegmenterWorker ─► seg_q ─► PipelineWorker ─► terminal
 Loopback (WASAPI) ─► AudioCapture ─┘   (32 ms,    (Silero VAD,                 │ STT  (faster-whisper)
   or FileSource (replay)               16 kHz      one state machine           │ Speaker ID (pyannote) ║ parallel
                                        mono)       per source)                 └ MT   (NLLB-200)
```

The pipeline uses threads and `queue.Queue`s. Each stage only exchanges the small data types in `core/types.py` (`AudioChunk`, `Segment`), so a model can be swapped without touching the other stages.

1. **Capture** (`audio/capture.py`): one PortAudio stream per device. The callback downmixes and resamples (soxr) the device audio (usually 48 kHz stereo) to 16 kHz mono float32. It stamps each chunk with session time and puts it on `chunk_q`. `audio/file_source.py` produces the same chunks from WAV files, paced in real time, for repeatable tests.
2. **Segmentation** (`audio/segmenter.py`): Silero VAD on 512-sample frames, with a separate state machine per source so mic and call never mix. A phrase is closed by:
   - 500 ms of silence, or
   - a forced cut at 8 s, made at the quietest frame in the last 1.5 s to avoid cutting mid-word, or
   - an idle timeout of 0.6 s when audio stops arriving. WASAPI loopback sends *no data at all* while nothing plays, so silence frames never come.

   Segments shorter than 0.4 s of speech are dropped, and 200 ms of padding is kept on each side.
3. **Pipeline worker** (`main.py`): a single GPU worker handles segments in arrival order. For each segment:
   - speaker identification is submitted to its own one-thread executor, so it runs **while Whisper decodes**;
   - speech-to-text runs (`stt/whisper.py`). A hallucination filter drops noise, YouTube-credit phrases, repetition loops, text in the wrong script and output with too little confidence;
   - the Armenian text is translated (`translate/nllb.py`);
   - the block is printed.

   One bad segment is logged and skipped, so it can't stop the session.
4. **Speaker ID** (`speakers/pyannote.py`):
   - Mic segments are always "Me".
   - Each system segment gets a WeSpeaker ResNet34 embedding, which is compared with the running centroid of every known speaker. If the cosine similarity is ≥ 0.5, the segment goes to that speaker and the centroid is updated.
   - Otherwise a segment ≥ 1.2 s creates a new speaker. A shorter one goes to the closest speaker without moving its centroid.
   - Segments under 0.6 s are not embedded; they reuse the last system speaker.

Speech-to-text and translation share one GPU worker on purpose. Both are CTranslate2 models on the same small GPU, and running them in parallel only makes each slower.

## 6. Models: what was chosen and why

Armenian is a low-resource language, so model choice decides most of the quality. Constraints: local only, fits a 4 GB GPU next to the other models, ≤ 5 s end-to-end. Full notes are in [`DECISION.md`](DECISION.md).

### Speech-to-text: `Chillarmo/whisper-large-v3-turbo-armenian` (CTranslate2 int8, beam 1)

| Candidate | Result |
|---|---|
| **Chillarmo turbo-armenian** ✅ | Whisper large-v3-turbo fine-tuned on Armenian (self-reported WER 15.3 % / CER 2.9 % on Common Voice). The turbo decoder (4 layers) is much faster than large-v3, and the int8 model is ~0.8 GB. Chosen: Armenian-specific training plus speed. |
| Whisper large-v3 (stock) | Works zero-shot (CER ~13.8 % FLEURS per the BuzzASR card). On the GTX 1650 with int8 it took ~1.2 s per phrase at beam 1 and ~1.4–1.5 s at beam 5, with nearly the same text. That is close to the 1.5 s STT budget, and it is not tuned for Armenian. Kept as `--stt-model large-v3`. |
| BuzzASR/armenian | Best published Armenian numbers (WER 11.9 % / CER 3.7 % FLEURS), but **rejected**: it uses its own tokenizer with `<|endoftext|>` at id 0. faster-whisper assumes the standard Whisper vocabulary, so it cannot decode it. It would need the much slower HF `transformers` runtime. `stt/convert_model.py` detects and refuses this layout. |
| SeamlessM4T v2, MMS / Omnilingual | Not tried: heavier with no CTranslate2 runtime (Seamless), or CTC output without punctuation, which hurts translation (MMS). |

Decoding settings:
- Language forced to `hy`.
- Each segment is decoded on its own (`condition_on_previous_text=False`, so one bad segment can't start a hallucination loop in the next).
- No temperature fallback: every fallback is a full extra decode and doubled latency in tests. The hallucination filter catches bad output instead.
- Output tokens are capped by audio length.
- Quiet segments are normalized up to +20 dB.

### Translation: NLLB-200 distilled 1.3B (CTranslate2 int8, beam 4)

| Candidate | Result |
|---|---|
| **NLLB-200 distilled 1.3B** ✅ | Built for low-resource pairs including `hye_Armn → eng_Latn`. ~1.4 GB in int8, fast in CTranslate2. License CC-BY-NC-4.0 (fine for this challenge, not for commercial use). |
| NLLB 600M / 3.3B | 600M is faster but weaker. 3.3B (~3.4 GB int8) does not fit next to Whisper on 4 GB. Both are one `--mt-model` switch away. |
| MADLAD-400 3B, local LLMs | Too large/slow for this GPU. LLMs may also "improve" or invent content. |
| Whisper `task="translate"` | Loses the Armenian text, which we need to print. Fine-tuned Armenian models also lose this ability. |

NLLB translates sentence by sentence, so a phrase with several sentences is split on `։ . ! ? …` and translated as one batch. Output length is capped, repetition loops are collapsed, and low-confidence translations are still printed (and flagged in `--debug`).

### Speaker ID: pyannote WeSpeaker embeddings + own online clustering

The full pyannote diarization pipeline works on whole recordings, and its labels can change on every rerun. Live output needs a label per segment that never changes afterwards. So only pyannote's embedding model (`wespeaker-voxceleb-resnet34-LM`, a few ms per segment) is used, together with the simple centroid clustering described in §5. `diart` (streaming diarization), SpeechBrain ECAPA and NeMo were considered. Embeddings plus our own clustering was the simplest option to explain and tune. The threshold can be tuned on your own clips with `python -m speakers.pyannote clips\*.wav`.

### VAD: Silero VAD

Small, accurate, runs on CPU, and widely used with Whisper. WebRTC VAD triggers falsely more often on compressed call audio, and an energy threshold fails with background noise.

### VRAM

| Component | Approx. VRAM |
|---|---|
| Whisper turbo-armenian (int8) | ~1 GB |
| NLLB 1.3B (int8) | ~1.4 GB |
| WeSpeaker ResNet34 (fp32) | < 0.2 GB |
| CUDA context / buffers | ~0.5–1 GB |
| **Total** | **~3–3.5 GB of 4 GB** |

`python -m bench.stt_speed <wav>` prints the real GPU, VRAM in use and a warning when Windows starts paging to shared memory.

## 7. Latency

Budget from the end of a phrase to the English line (PLAN.md §4):

| Stage | Target |
|---|---|
| VAD silence wait | 0.5 s |
| Speech-to-text (≤ 8 s audio) | ≤ 1.5 s |
| Translation | ≤ 0.5 s |
| Speaker embedding | runs in parallel with speech-to-text |
| Queue / print | ≤ 0.3 s |
| **Total** | **≈ 3 s, leaving ~2 s headroom** |

How it is kept down:
- forced cut at 8 s
- int8 models and warm-up before listening starts
- greedy decoding for speech-to-text
- no temperature fallback
- output length caps
- speaker ID running in parallel

Every phrase's latency is measured (`--debug`), and on exit the program prints the mean, p95, max and the number of phrases over 5 s.

**Measured end-to-end (replay of `recordings/`, GTX 1650):** _fill in from the summary line of a `--debug` replay run: mean / p95 / max._

If both sides talk at the same time, segments wait for the single GPU worker. A long burst of overlapping speech can push some phrases over 5 s (see `queue` in `--debug`).

## 8. Testing

| What | How |
|---|---|
| Unit tests | `python -m pytest tests/`: segmenter state machine with a fake VAD (pauses, forced cut, idle timeout, stream gaps); translation text handling and the engine with a fake model (sentence splitting, Armenian punctuation, repetition collapse, confidence flags). No GPU or models needed. |
| Audio capture | `python -m audio.capture --record 10` → listen back to `mic.wav` / `system.wav`, check the level meter and overflow count |
| Speech-to-text speed and output | `python -m bench.stt_speed recordings\me\mic.wav --models models/whisper-turbo-hy large-v3 --beams 1 5`: time per phrase, VRAM, text side by side |
| STT on a file | `python -m stt.whisper recordings\me\mic.wav --segment` |
| Translation | `python -m translate.nllb "Բարև, ինչպե՞ս ես։"` or `--wav <file>` for STT → MT on a recording |
| Speaker threshold | `python -m bench.cut_clips recordings\other\system.wav other` → `python -m speakers.pyannote clips\*.wav` prints the similarity matrix and labels |
| End-to-end | `python main.py --system-file ... --mic-file ... --debug` (replay mode, real-time pacing, latency summary at the end) |
| Live | A real call with 2+ Armenian speakers on the call side |

Test recordings are my own (in `recordings/` and `clips/`, git-ignored).

## 9. Project structure

```
main.py                  orchestrator: CLI, model loading, threads, output, latency stats
core/types.py            AudioChunk, Segment, SessionClock (shared data types)
audio/capture.py         WASAPI mic + loopback capture, --list / --record helper
audio/file_source.py     WAV replay as a fake live source
audio/segmenter.py       Silero VAD segmentation (per-source state machine + worker thread)
stt/whisper.py           faster-whisper wrapper + hallucination filter
stt/convert_model.py     HF Whisper fine-tune -> CTranslate2 (+ tokenizer/preprocessor checks)
translate/nllb.py        NLLB-200 wrapper (sentence split, batching, post-filter)
translate/convert_model.py  NLLB -> CTranslate2 (+ tokenizer saved into the folder)
speakers/pyannote.py     WeSpeaker embeddings + online clustering, threshold-tuning tool
bench/stt_speed.py       speech-to-text speed/VRAM benchmark per model, compute type and beam
bench/cut_clips.py       cut recordings into clips for speaker-threshold tuning
tests/                   pytest unit tests (no GPU needed)
PLAN.md, DECISION.md     plan and decision log
chats/                   exported AI chat transcripts
```

## 10. Known limitations

- **Windows only for live capture** (WASAPI loopback). Replay mode is cross-platform.
- **Speech-to-text quality** is the weakest link. Conversational, fast or overlapping Armenian, names and English loanwords are often misrecognized, and translation errors follow. Very short replies ("Այո", "Հա") can be dropped by the minimum-length (0.4 s) and confidence filters.
- **Accuracy was not measured with WER/CER on reference transcripts.** The models were chosen from published numbers, compatibility and speed, and by reading their output on my recordings.
- **One label per segment**: if two people on the call speak inside the same phrase, the whole phrase gets one speaker. Short segments (< 0.6 s) reuse the last speaker. Similar voices, or the same voice over a bad connection, can be merged or split. The threshold (0.5) is based on a small set of my own clips and may need adjusting for other voices or call apps (`--speaker-threshold`).
- **No echo handling**: with loudspeakers the call is also transcribed as "Me".
- **Phrase-level, not streaming**: nothing is printed until a pause or the 8 s cut, so a phrase appears as one block.
- **One GPU worker**: simultaneous speech from both sides is processed one phrase after another and can exceed 5 s under heavy overlap.
- Translation is sentence-level with no context from previous phrases. Pronouns and ellipsis may come out wrong.
- NLLB weights are non-commercial (CC-BY-NC-4.0).

## 11. What I would improve with more time

1. **Proper benchmark** with hand-written reference transcripts: WER/CER for turbo-armenian vs large-v3, and chrF / manual ratings for NLLB 1.3B vs 600M. Then pick the models from those numbers.
2. **Run BuzzASR** via `transformers` (or by remapping its tokenizer for CTranslate2) to see if its better accuracy is worth the latency.
3. **Streaming partial results**: show a draft line while someone is still speaking and replace it when the phrase closes.
4. **Better speaker ID**: overlap-aware streaming diarization (`diart`), speaker-change detection inside a segment, and threshold tuning on more voices and call codecs.
5. **Context-aware translation**: pass the previous phrase as context, or try a small local LLM for fluency on a bigger GPU.
6. Echo cancellation, so loudspeakers can be used.
7. Priority scheduling when both sides speak at once (oldest `t_end` first across sources), and dropping hopelessly late segments instead of delaying everything.
