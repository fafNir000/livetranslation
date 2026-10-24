# TEST — how to test each package and script

How to check every part of the pipeline on its own, then all of it together. Run the commands from the project root with the venv active (`.venv\Scripts\activate`). Always use `python -m package.module`, not `python package\module.py`, so imports resolve. This matters most for `speakers.pyannote`, whose file name would otherwise shadow the real `pyannote` package.

Steps are ordered from "no hardware needed" to "full live call". If a step fails, fix it before going on: later steps depend on earlier ones.

| # | What | Needs GPU | Needs models | Needs live audio |
|---|---|---|---|---|
| 0 | Environment | – | – | – |
| 1 | Unit tests (`tests/`) | – | – | – |
| 2 | Audio capture (`audio/capture.py`) | – | – | yes |
| 3 | Segmenter (`audio/segmenter.py`) | – | Silero (auto) | – (WAV) |
| 4 | Model conversion (`stt/`, `translate/convert_model.py`) | – | downloads | – |
| 5 | Speech-to-text (`stt/whisper.py`, `bench/stt_speed.py`) | yes | Whisper | – (WAV) |
| 6 | Translation (`translate/nllb.py`) | yes | NLLB | – |
| 7 | Speaker ID (`speakers/pyannote.py`, `bench/cut_clips.py`) | optional | pyannote | – (WAV) |
| 8 | End-to-end replay (`main.py` + `audio/file_source.py`) | yes | all | – (WAV) |
| 9 | Live call (`main.py`) | yes | all | yes |

---

## 0. Environment

```bat
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"
python -c "import faster_whisper, pyannote.audio, silero_vad, soxr, soundfile, pyaudiowpatch; print('imports ok')"
```

**Pass:** Python 3.11, `True` and the GPU name, a CUDA device count of 1 or more, `imports ok`.

If `torch.cuda.is_available()` is `False`, the CPU build of torch was installed. Reinstall it from the CUDA index (README §2). If CTranslate2 fails with `cudnn_ops64_9.dll not found`, `import torch` must run before `faster_whisper`. `main.py` already does this.

## 1. Unit tests (no GPU, no models)

```bat
python -m pytest tests/ -v
```

| File | Covers |
|---|---|
| `tests/test_segmenter.py` | Segmenter state machine with a fake VAD: phrase closed by a pause, two phrases, short blip dropped, forced cut at 8 s, odd chunk sizes, idle timeout, gap in the stream |
| `tests/test_translate.py` | Text handling (`prepare_source`, `split_sentences`, `collapse_repeats`, `clean_output`, Armenian ratio / flags) and `NllbTranslator` with a fake model (single sentence, batched sentences, empty input, low confidence, `translate_stt`) |

**Pass:** all 16 tests green.

Not covered by unit tests: online speaker clustering, resampling in capture, the STT hallucination filter. Steps 5 and 7 test those by hand.

## 2. Audio capture — `audio/capture.py`

List devices (the same list as `python main.py --list-devices`):

```bat
python -m audio.capture --list
```

Record 10 s from both sources. Play something in the call app or a video, and speak into the mic at the same time:

```bat
python -m audio.capture --mic 1 --loopback 17 --record 10 --out recordings
```

Use the indices from `--list`. Without `--mic` / `--loopback`, the Windows defaults are used.

**Pass:**

- the level meter moves for both `mic` and `system` while recording;
- `recordings\mic.wav` and `recordings\system.wav` play back clean, at normal speed and pitch (pitch or speed shifted means a resampling problem);
- no "input overflows" warning at the end;
- with nothing playing, the `system` level stays empty. This is expected: WASAPI loopback sends no data in silence.

## 3. Segmenter — `audio/segmenter.py`

The unit tests (step 1) cover the logic. To see real phrase boundaries on a recording, cut it with Silero VAD and print the segments:

```bat
python -m stt.whisper recordings\me\mic.wav --model models/whisper-turbo-hy --segment
```

Each printed line starts with `[start-end]` in seconds.

**Pass:** one line per spoken phrase, no segment longer than 8 s, and words are not cut in half at boundaries (listen to the WAV at the printed times if in doubt).

## 4. Model conversion — `stt/convert_model.py`, `translate/convert_model.py`

Convert once (needs network), then check that the folders are complete:

```bat
python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy
python -m translate.convert_model

:: check only, no download
python -m stt.convert_model --fix models/whisper-turbo-hy
python -m translate.convert_model --check models/nllb-200-distilled-1.3B
```

**Pass:** every listed file is `ok`. For Whisper, `mel bins: 128` and `tokenizer: ok`. For NLLB, `model.bin` is about 1.4 GB and `check: ok`.

Negative test: `python -m stt.convert_model --fix models/buzzasr-hy` should report the tokenizer as `NOT USABLE` (non-standard vocabulary). This is why BuzzASR was rejected (DECISION.md, Stage 3).

## 5. Speech-to-text — `stt/whisper.py`, `bench/stt_speed.py`

Note: the standalone default model of `stt.whisper` is stock `large-v3`. Pass `--model models/whisper-turbo-hy` to test the model the program actually uses.

Transcribe a recording phrase by phrase:

```bat
python -m stt.whisper recordings\me\mic.wav --model models/whisper-turbo-hy --segment -v
```

**Pass:** readable Armenian text per phrase. Silence and noise show up as `-- dropped: <reason>`, not as invented text.

Speed and VRAM on this machine, for the chosen model against stock large-v3:

```bat
python -m bench.stt_speed recordings\me\mic.wav --models models/whisper-turbo-hy large-v3 --compute-types int8_float32 --beams 1 5
```

**Pass:** for `models/whisper-turbo-hy`, `int8_float32`, beam 1, the summary shows a mean time per phrase of at most 1.5 s, and there is no "Windows will page to shared memory" warning. The environment block at the top should show the real GPU and about 4 GiB of dedicated VRAM.

Hallucination filter check: run `stt.whisper` (without `--segment`) on a WAV of silence or background noise only. Every result should be dropped.

## 6. Translation — `translate/nllb.py`

Typed phrases (text only, no STT):

```bat
python -m translate.nllb "Բարև, ինչպե՞ս ես։" "Լավ եմ, շնորհակալություն։ Իսկ դու՞։"
```

From a file (one phrase per line, UTF-8):

```bat
python -m translate.nllb --file phrases_hy.txt
```

On real STT output (Whisper, then NLLB on each phrase of a recording):

```bat
python -m translate.nllb --wav recordings\me\mic.wav --stt-model models/whisper-turbo-hy
```

**Pass:** understandable English. A two-sentence input gives two translated sentences, the second one not dropped. The timing summary at the end shows a mean of at most about 0.5 s per phrase. Low-confidence outputs are marked, not hidden.

Optional comparison with the smaller model, if it has been converted: `--model models/nllb-200-distilled-600M`.

## 7. Speaker ID — `bench/cut_clips.py`, `speakers/pyannote.py`

Cut recordings of different people into 3 s clips. Clips named `me_*` go through the mic path and are labelled "Me", like in the real pipeline:

```bat
python -m bench.cut_clips recordings\me\mic.wav me
python -m bench.cut_clips recordings\other\system.wav other
```

Embed all clips, cluster them and print the similarity matrix:

```bat
python -m speakers.pyannote clips\me_1.wav clips\me_2.wav clips\other_1.wav clips\other_2.wav clips\other_3.wav
python -m speakers.pyannote clips\other_1.wav clips\other_2.wav --threshold 0.6
```

**Pass:**

- `me_*` clips are labelled `Me`;
- clips of the same person get the same `Speaker N`, and different people get different numbers;
- in the matrix, same-person pairs score clearly above 0.5 and different-person pairs below it;
- each embedding takes only a few ms on GPU.

If different people merge into one speaker, raise the threshold. If one person splits into several, lower it. Pass the value you choose to `main.py` as `--speaker-threshold`.

## 8. End-to-end replay — `main.py` with `audio/file_source.py`

Replay WAV files through the full pipeline in real time, as if they were live:

```bat
python main.py --system-file recordings\other\system.wav --mic-file recordings\me\mic.wav --debug --offline
```

Variants:

```bat
:: one side only
python main.py --system-file recordings\other\system.wav --debug

:: both sides overlapping, to stress the single GPU worker
python main.py --system-file recordings\other\system.wav --mic-file recordings\me\mic.wav --system-offset 0 --debug

:: speaker ID switched off (fallback path)
python main.py --system-file recordings\other\system.wav --no-speaker-id --debug
```

**Pass:**

- each phrase prints as `[hh:mm:ss] Label` / `HY:` / `EN:`, with mic phrases labelled `Me` and system phrases `Speaker N`;
- the `[latency ... ]` line shows where time goes. Expect `vad` about 0.5 s, `stt` at most 1.5 s, `mt` at most 0.5 s, and `queue` near 0 unless both sides overlap;
- the summary on exit shows p95 under 5 s and `0 over 5 s`.

Copy that summary line into README §7.

`--offline` also proves that no model needs the network. It fails if the pyannote model was never cached, so run once without `--offline` first.

## 9. Live call — `main.py`

```bat
python main.py --list-devices
python main.py --mic 1 --loopback 17 --offline --debug
```

Checklist:

- [ ] "Listening..." appears after model loading, and there are no errors;
- [ ] your own speech appears as `Me`;
- [ ] two or more people on the call side get stable, different `Speaker N` labels for the whole session;
- [ ] Armenian is recognised and the English is understandable;
- [ ] latency stays under 5 s; check the summary after Ctrl+C;
- [ ] it runs for 10 minutes or more without crashing;
- [ ] short replies ("Այո", "Հա") are noted: they may be dropped, which is a known limitation;
- [ ] Ctrl+C stops cleanly and prints the summary.

Also try the failure cases you may be asked about. With loudspeakers instead of headphones, the call appears twice, once as `Me` (no echo handling, out of scope). Talking over the other side shows the `queue` time grow.

---

## Quick run (about 5 minutes)

```bat
python -m pytest tests/ -q
python -m audio.capture --mic 1 --loopback 17 --record 10 --out recordings
python main.py --system-file recordings\other\system.wav --mic-file recordings\me\mic.wav --offline --debug
python main.py --mic 1 --loopback 17 --offline
```
