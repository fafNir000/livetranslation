**PROMPT**

```
Task
Build a Python program that listens to a live conversation, transcribes spoken Armenian, and prints an English translation in the terminal within 5 seconds, labeled by speaker (the full task description and requirements can be found in the document `Mathrix_AI_Coding_Challenge.docx`).
Audio capture and segmentation into phrases have already been implemented (see the `audio` and `core` folders). Speech recognition and translation are not yet included in these folders.
Based on `PLAN.MD` (you may also refer to `DECISION.MD` and `requirements.txt` if needed), please write the `stt/whisper.py` file for me.
```

**RESPONSE**

Probably a venv inside. Let me filter.Now I have the full context (brief, plan, decisions, types, segmenter, file replay). Let me write the module.Now a package `__init__.py`, then a real test run against your recordings with a small model on CPU.Filter logic works. Now an end-to-end run of the engine and CLI on your recordings (small model, CPU, just to exercise the code path).PyPI isn't reachable from this sandbox, so I can't load a real Whisper model here. I'll test the engine logic instead with a stubbed `faster_whisper` module.A whole utterance that's just a repetition loop shouldn't pass as "ու". I'll reject the result when collapsing removes most of the words.Works as intended. Saving both files into your project:I've written `stt/whisper.py` and a one-line `stt/__init__.py` so `stt` is a package like `audio`. Both are in `C:\Chris\livetranslation\stt\`. It has not been run with a real Whisper model yet: this sandbox couldn't download packages, so the first real run will be on your laptop.

**How it works**

- **Settings live in one config object.** `WhisperConfig` holds every setting, and the model is just one of them. It can be `large-v3`, `large-v3-turbo`, or a folder with a converted fine-tune such as BuzzASR or Chillarmo. The conversion command is in the file's top comment. So the benchmark in DECISION.md can swap models without code changes.
- **The engine.** `WhisperSTT` loads the model once and runs a warm-up pass. Then `transcribe_segment(segment)` returns an `SttResult`: the Armenian text, an `ok` flag, the reason if it was rejected, confidence numbers, and how long the call took (for the latency log).
- **Settings chosen for the 5 s limit.** Language is forced to Armenian, each segment is decoded on its own (no carry-over from the previous one), timestamps are off, Whisper's own VAD is off because your segmenter already cut on speech, and there are only 3 retry temperatures.
- **Hallucination filter.** It rejects:
  - empty or very low-confidence output, or output the model itself flags as "no speech"
  - known YouTube credit phrases ("thanks for watching", "subscribe", Amara.org). A plain "Շնորհակալություն" still passes.
  - text that isn't in Armenian script
  - output that loops on one phrase
  - more text than the audio could hold

  Shorter repeats inside real speech are collapsed instead of rejected. The thresholds are first guesses to tune on your recorded clips.
- **Quick check from the terminal.** `python -m stt.whisper recordings/system.wav --segment` cuts the file into phrases with Silero first, then prints each phrase with its decode time and confidence. Dropped phrases are listed with the reason.

**What I tested here:** the filter and repeat-collapsing on Armenian, Russian and credit-phrase samples, plus the whole transcribe path with a stand-in for faster-whisper. Normal speech passed, and repetition loops, wrong script, silence, too-short audio and low confidence were all rejected. I did not unit-test the `--segment` path.

**Next:** run the terminal check on your recordings on the GPU to see real speed and accuracy. Those numbers will show whether the filter thresholds need adjusting.

**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m stt.whisper recordings/system.wav --segment
C:\Chris\livetranslation\.venv\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\lekalaprint\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING huggingface_hub.utils._http: Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
C:\Chris\livetranslation\.venv\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\lekalaprint\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
```

**RESPONSE**

These are all warnings, not errors. The program is most likely still running and downloading Whisper large-v3, which is about 3 GB. faster-whisper turns off the download progress bar, so it looks frozen. Leave it running. Once the model is loaded and warmed up you'll see one line per phrase.

The warnings:

- **Symlinks:** Windows without Developer Mode stores the cache by copying files instead of linking them. It works, it just uses more disk. You can ignore it or turn on Developer Mode.
- **No HF_TOKEN:** downloads are slower without one, but they work. You'll need a token for pyannote anyway, so put `HF_TOKEN=...` in your `.env`.

A silent wait like this is confusing, so I'll make the terminal check print status messages and hide the symlink warning:Your run is most likely still downloading the model, so let it keep going. Once the model is cached, later runs only need to load it, which should take a few seconds.

I've updated `stt/whisper.py` in your project. The terminal check now prints "Loading Whisper… (first run downloads ~3 GB)" before it starts, then "Model ready in X s" once the model is loaded. It also hides the symlink warning. The change only affects the terminal check, not how the engine works.

To see the download moving, watch `C:\Users\lekalaprint\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3` grow in Explorer.

If there's still no output after the folder reaches about 3 GB, or the command ends with nothing printed, paste whatever the terminal shows and I'll dig in. A likely cause would be a missing CUDA or cuDNN library error.

For the interview, set `local_files_only=True` once all models are cached. That way the program never waits on the network.

**PROMPT**

```
The text does not match what I said, and the model takes too long to provide a response (the latency does not meet the requirements).
(.venv) PS C:\Chris\livetranslation> python -m stt.whisper recordings/mic.wav --segment
[  1.86-  3.78s] 12.32s  lp=-0.43  HY: Հավովում բաշէվ ես էս
[  5.41-  6.34s] 10.49s  lp=-0.23  HY: Չպես սեք։
```

**RESPONSE**

Now the code changes. First, quiet-input gain and a clear GPU check in the engine:Now I'll pull the offline segmentation out into a reusable function, so a benchmark script can use it too:Now a speed/accuracy check script. It reports what the GPU actually is, then times each model setup on your clip:A `bench/stt_speed.py` already exists in the session outputs. I didn't write it in this conversation, so I'll read it before deciding anything:Testing with stubbed faster-whisper/ctranslate2 (real models can't be downloaded here):Everything runs. Saving to your project:There are two separate problems, and I've added code for both. I tested it here only against stand-ins for faster-whisper and the GPU library, because model downloads are blocked in this sandbox. So the real numbers have to come from your laptop.

**1. Latency: about 12 s for a 2 s phrase**

Whisper large-v3 on a working NVIDIA GPU should take well under 1 s for that, so the GPU isn't doing its job. My best guess: the "11.9 GB" Windows reports may be dedicated VRAM plus shared system RAM. On a laptop GPU with 4–6 GB of real VRAM, the 3 GB model spills into system RAM and runs about 10× slower. The other common cause is the laptop running on battery or in power-saving mode. If the shared-RAM guess is right, the VRAM budget in PLAN.md (about 8 GB) doesn't hold, and that changes the model choices.

To find out, please run this and paste the whole output:

```
python -m bench.stt_speed recordings/mic.wav
```

It's a new script. It first prints the GPU name, the real dedicated VRAM and the power state. Then it times large-v3 and large-v3-turbo at beam 1 and beam 5 on each phrase, with the text, and ends with an OK / TOO SLOW summary against the 1.5 s budget. Run it with the charger plugged in.

**2. Accuracy**

- **Your mic recording is very quiet.** I measured it: it peaks at about −20 dBFS, and speech is only about 12 dB above the background noise. That hurts Whisper. `stt/whisper.py` now boosts each phrase to a normal level, by at most +20 dB so background noise isn't turned into "speech". A small part of the fix is on your side: raise the mic level in Windows sound settings.
- **Stock Whisper is weak on Armenian.** DECISION.md expected this. The real fix is an Armenian fine-tune. Chillarmo turbo-armenian should help with speed too, since it's about half the size of large-v3. A new script, `stt/convert_model.py`, converts a fine-tune so faster-whisper can run it:
  ```
  python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy
  python -m bench.stt_speed recordings/mic.wav --models large-v3-turbo models/whisper-turbo-hy
  ```
  I couldn't reach Hugging Face from here, so the conversion itself hasn't been tried. Make sure `models/` is in your `.gitignore`.

`stt/whisper.py` also now stops with a clear error if the GPU library can't see your GPU.

Once I see the benchmark output, I can tell which it is, a GPU or power problem or simply a model too big for this card, and pick the default model and beam size.

**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 8.06, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [float16] ===
  load+warm-up 19.7 s, VRAM for this model ~3901 MiB
  beam 1 [ 1.86- 3.78s] 10.58s lp=-0.45 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  9.92s lp=-0.28 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s] 12.09s lp=-0.43 gain=2.18  Հավովում բաշէվ ես էս
  beam 5 [ 5.41- 6.34s] 10.48s lp=-0.20 gain=6.07  Չպես սեք։

=== large-v3-turbo  [float16] ===
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  load+warm-up 121.9 s, VRAM for this model ~2205 MiB
  beam 1 [ 1.86- 3.78s] 13.42s lp=-1.55 gain=2.18 [FALLBACK]  -- dropped: low confidence (avg_logprob -1.55) (raw: "Aho, bosh, e, what's this?")
  beam 1 [ 5.41- 6.34s] 17.97s lp=-0.24 gain=6.07 [FALLBACK]  -- dropped: repetitive (compression ratio 11.79) (raw: 'էպպսկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկկ')
  beam 5 [ 1.86- 3.78s] 12.79s lp=-1.65 gain=2.18 [FALLBACK]  -- dropped: low confidence (avg_logprob -1.65) (raw: 'Aho, bosh, hãy subscribe to this.')
  beam 5 [ 5.41- 6.34s]  7.98s lp=-0.87 gain=6.07  չպշշը եկկկկկկկկկ

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 float16           1   10.25   10.58  TOO SLOW
large-v3                                 float16           5   11.28   12.09  TOO SLOW
large-v3-turbo                           float16           1   15.69   17.97  TOO SLOW
large-v3-turbo                           float16           5   10.39   12.79  TOO SLOW
```

**RESPONSE**

The key fact: the GPU is a **GTX 1650 with 4 GB**. That's a very different card from the planned "11.9 GB" one. The 11.9 GB Windows reported was dedicated plus shared system RAM, as suspected. Let me check what this means for the code.Line 492 contains code I didn't write, so the file changed since my last commit. Let me compare it with the copy on your laptop before editing:The laptop copy matches the session copy, so nothing is lost. The extra line is a small fallback marker in the terminal check. It's harmless, so I'll keep it. Now the changes for the GTX 1650:Now the benchmark defaults. I'm adding int8 setups and a warning when VRAM is nearly full:I've changed `stt/whisper.py` and `bench/stt_speed.py` for your actual GPU, a GTX 1650 with 4 GB. The changes are untested on real hardware: I could only run them here against stand-ins, so the next benchmark run will tell whether they work.

**Why it was slow:** large-v3 in float16 took up 3.9 of the card's 4 GB. When VRAM is that full, Windows moves part of it into system RAM and everything runs about 10× slower, which explains the 10–12 s. The GTX 16 series also lacks the hardware that makes float16 fast, so it buys no speed on this card. Turbo was slower still because its bad first pass triggered extra decodes: the "[FALLBACK]" marks.

**Why the text was so bad:** stock Whisper is weak on Armenian, and the turbo model is worse. Its output drifted into English ("subscribe") and repeated letters ("կկկկ…"). Running out of memory may have hurt quality too, which the next run will show.

**What changed:**

- **8-bit model (int8) is now the default.** large-v3 should need about 1.6 GB instead of 3.1 GB, so it fits on the card.
- **One decode pass per phrase.** Retries are off by default, since each one is a full extra decode. Bad output is caught by the filter instead.
- **Output length is capped by phrase length.** A looping decoder now stops early instead of producing hundreds of tokens.
- **New filter rule:** a letter repeated 5 or more times ("եկկկկկ") is rejected. That one slipped through last time.
- **The benchmark now tests int8 by default.** It also warns when VRAM is more than 85% full.

Please run the same command again and paste the output:

```
python -m bench.stt_speed recordings/mic.wav
```

**What to expect:** Whisper always processes a fixed 30 s window, however short the phrase. So on this card even the int8 model may stay around 1–2 s per phrase, against the 1.5 s budget. The turbo models, including the Armenian one, only speed up the text-generating half, so the gain is limited. Whatever the result, these numbers will settle which model to use.

**This also affects the rest of the plan:** PLAN.md assumed about 8 GB of VRAM, and only 4 GB is available. Whisper has to keep the GPU. Translation should be NLLB-600M in int8 or run on the CPU, and the pyannote speaker model and Silero should run on the CPU. I can update PLAN.md and DECISION.md once we have the numbers.

**One more thing:** the copy of `stt/whisper.py` on your laptop had a change I hadn't made, a small fallback marker in the terminal check. I compared the files before editing, kept the marker, and made sure nothing on your side was overwritten.

**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 8.22, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [float16] ===
  load+warm-up 17.8 s, VRAM for this model ~3901 MiB
  beam 1 [ 1.86- 3.78s] 10.53s lp=-0.45 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  9.89s lp=-0.28 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s] 12.06s lp=-0.43 gain=2.18  Հավովում բաշէվ ես էս
  beam 5 [ 5.41- 6.34s] 10.45s lp=-0.20 gain=6.07  Չպես սեք։

=== large-v3-turbo  [float16] ===
  load+warm-up 11.3 s, VRAM for this model ~2205 MiB
  beam 1 [ 1.86- 3.78s]  7.92s lp=-0.63 gain=2.18  -- dropped: not Armenian script (0% Armenian letters) (raw: 'Aho, bosh, ַַַַַַַַַַַַַַַַַַ')
  beam 1 [ 5.41- 6.34s]  7.75s lp=-0.99 gain=6.07  էպպսսկ եպպտանց էպ
  beam 5 [ 1.86- 3.78s]  8.21s lp=-0.59 gain=2.18  -- dropped: repetition loop (9 -> 2 words after collapsing) (raw: 'հով հով հով հով հով հով հով հով հո')
  beam 5 [ 5.41- 6.34s]  7.93s lp=-0.81 gain=6.07  -- dropped: character loop ('կկկկկկկկ'...) (raw: 'չպշշը եկկկկկկկկ')

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 float16           1   10.21   10.53  TOO SLOW
large-v3                                 float16           5   11.26   12.06  TOO SLOW
large-v3-turbo                           float16           1    7.84    7.92  TOO SLOW
large-v3-turbo                           float16           5    8.07    8.21  TOO SLOW
```

**RESPONSE**

The new filter rules are active, but the benchmark still ran **float16** and didn't print the VRAM warning. So the `bench/stt_speed.py` that ran isn't the one I saved. Let me check the laptop copy:That run still used float16, the setup that was already too slow. Your copy of `bench/stt_speed.py` had been overwritten with an older version, and my int8 change is in that file. I've saved the right version again.

Please run this. The flags pick the 8-bit (int8) setups even if the file gets reverted again:

```
python -m bench.stt_speed recordings/mic.wav --compute-types int8_float32 int8_float16
```

If you have another Claude session open on this folder, or ran a git checkout or restore, that's probably what overwrote the file. Let me know, so we don't keep overwriting each other's work.

What this run does show:

- **The new decode settings work.** "[FALLBACK]" is gone, and turbo dropped from 10–18 s to about 8 s per phrase, mostly from removing the extra decode passes.
- **The new filter rules work.** All three garbage outputs from turbo were rejected (wrong script, repeated word, repeated letter).
- **Memory isn't the whole story.** Turbo uses only 2.2 of the 4 GB, so it shouldn't be swapping, yet it still takes about 8 s. So float16 itself looks slow on this card, which makes the int8 run the important one.

Even in the best case, Whisper large-v3 on a GTX 1650 likely needs about 1 s per phrase. It always processes a fixed 30 s window, however short the phrase. If int8 lands around 1–1.5 s we're within budget. If it's still far above, we'll need a smaller model, and I'll go through the options for that.
**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav --compute-types int8_float32 int8_float16
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 8.57, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [int8_float32] ===
  load+warm-up 9.2 s, VRAM for this model ~2077 MiB
  beam 1 [ 1.86- 3.78s]  1.28s lp=-0.42 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  1.13s lp=-0.25 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s]  1.54s lp=-0.42 gain=2.18  Հավովով բաշէվ սզս։
  beam 5 [ 5.41- 6.34s]  1.25s lp=-0.22 gain=6.07  Չպես սեք։

=== large-v3  [int8_float16] ===
  load+warm-up 8.2 s, VRAM for this model ~2047 MiB
  beam 1 [ 1.86- 3.78s]  2.22s lp=-0.45 gain=2.18  Հավով բաշտ էվ սզս։
  beam 1 [ 5.41- 6.34s]  2.07s lp=-0.26 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s]  2.82s lp=-0.40 gain=2.18  Հավովում բաշէվ ես էս։
  beam 5 [ 5.41- 6.34s]  2.34s lp=-0.21 gain=6.07  Չպես սեք։

=== large-v3-turbo  [int8_float32] ===
  load+warm-up 5.3 s, VRAM for this model ~1183 MiB
  beam 1 [ 1.86- 3.78s]  0.94s lp=-0.83 gain=2.18  հով ապսել սսս ասս
  beam 1 [ 5.41- 6.34s]  0.86s lp=-0.92 gain=6.07  էտպսսսկ
  beam 5 [ 1.86- 3.78s]  0.99s lp=-0.63 gain=2.18  -- dropped: repetition loop (9 -> 2 words after collapsing) (raw: 'հով հով հով հով հով հով հով հով հո')
  beam 5 [ 5.41- 6.34s]  0.93s lp=-0.84 gain=6.07  էտպսսկ էտպսկ էտպ

=== large-v3-turbo  [int8_float16] ===
  load+warm-up 5.1 s, VRAM for this model ~1151 MiB
  beam 1 [ 1.86- 3.78s]  1.86s lp=-0.79 gain=2.18  հով ապշպ ասպ ասպ ասպսիս ասպ�
  beam 1 [ 5.41- 6.34s]  1.82s lp=-0.98 gain=6.07  էպպսսկ եպպտակ էպտ
  beam 5 [ 1.86- 3.78s]  1.97s lp=-0.62 gain=2.18  -- dropped: repetition loop (9 -> 2 words after collapsing) (raw: 'հով հով հով հով հով հով հով հով հո')
  beam 5 [ 5.41- 6.34s]  1.83s lp=-0.86 gain=6.07  չպշշշըք

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 int8_float32      1    1.20    1.28  OK
large-v3                                 int8_float32      5    1.40    1.54  TOO SLOW
large-v3                                 int8_float16      1    2.15    2.22  TOO SLOW
large-v3                                 int8_float16      5    2.58    2.82  TOO SLOW
large-v3-turbo                           int8_float32      1    0.90    0.94  OK
large-v3-turbo                           int8_float32      5    0.96    0.99  OK
large-v3-turbo                           int8_float16      1    1.84    1.86  TOO SLOW
large-v3-turbo                           int8_float16      5    1.90    1.97  TOO SLOW
(.venv) PS C:\Chris\livetranslation>
```

**RESPONSE**

int8_float32 fixed the speed: large-v3 now decodes a phrase in about 1.2 s, down from about 10 s. Before I set beam 1 as the default, let me check that `whisper.py` on your laptop hasn't changed again:Speed is solved: large-v3 in 8-bit (int8_float32) decodes a phrase in about 1.2 s, down from about 10 s. Accuracy is not solved. The Armenian text is still wrong, and that's the model, not the hardware.

**Speed and memory on your GTX 1650**

| Setup | Time per phrase | VRAM | Text quality |
|---|---|---|---|
| large-v3, int8_float32, beam 1 | about 1.2 s ✅ | 2.1 GB | about the same as float16, still wrong |
| large-v3, int8_float32, beam 5 | about 1.4–1.5 s (borderline) | 2.1 GB | barely different from beam 1 |
| large-v3-turbo, int8_float32 | about 0.9 s ✅ | 1.2 GB | unusable on Armenian |
| any int8_float16 setup | 1.8–2.8 s ❌ | similar | similar |

- **The fix was the memory format.** float16 is slow on GTX 16-series cards. With int8 plus float32 the card is about 8× faster, and int8_float16 is about twice as slow as int8_float32. So the default should stay int8_float32.
- **I changed the default beam size to 1** in `stt/whisper.py`, since beam 5 didn't improve the text.
- **The whole chain should still fit in 5 s:** about 0.5 s waiting for the pause, 1.2 s speech recognition, an estimated 0.3–0.5 s translation, plus the printing.
- **That leaves about 1.9 GB of VRAM** for translation. That's enough for NLLB-600M in int8 (about 0.7 GB), or the 1.3B model in int8 (about 1.4 GB). The pyannote speaker model can run on the CPU.

**Next: try the Armenian fine-tunes**

Stock Whisper doesn't handle Armenian well enough. The fine-tunes listed in DECISION.md are the planned fix, and both should be fast enough on this card:

```
python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy --quantization int8_float32
python -m stt.convert_model BuzzASR/armenian models/buzzasr-hy --quantization int8_float32
python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy models/buzzasr-hy --compute-types int8_float32 --beams 1 5
```

The turbo fine-tune is the most promising: if its Armenian is good, it's fast enough to run at beam 5. The conversion script hasn't been run yet, because I couldn't reach Hugging Face from here. BuzzASR uses its own tokenizer, so it's the more likely of the two to fail. If either errors, paste the message.

Your mic recording is also very quiet, so turn up the mic level in Windows before recording more test clips. Two short phrases aren't enough to choose a model, so plan to record the 10–15 clips from PLAN.md.

What did you actually say in the two phrases in `mic.wav`? I need that to judge how far off each model is.

**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy models/buzzasr-hy --compute-types int8_float32 --beams 1 5 
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 5.68, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [int8_float32] ===
  load+warm-up 11.8 s, VRAM for this model ~2077 MiB
  beam 1 [ 1.86- 3.78s]  1.28s lp=-0.42 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  1.13s lp=-0.25 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s]  1.54s lp=-0.42 gain=2.18  Հավովով բաշէվ սզս։
  beam 5 [ 5.41- 6.34s]  1.25s lp=-0.22 gain=6.07  Չպես սեք։

=== models/whisper-turbo-hy  [int8_float32] ===
  load failed: ValueError: Invalid input features shape: expected an input with shape (1, 128, 3000), but got an input with shape (1, 80, 3000) instead

=== models/buzzasr-hy  [int8_float32] ===
  load failed: ValueError: 'models/buzzasr-hy' is English-only; an Armenian-capable model is needed

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 int8_float32      1    1.21    1.28  OK
large-v3                                 int8_float32      5    1.39    1.54  TOO SLOW
```

**RESPONSE**

Both conversions produced folders that faster-whisper can't use as they are. Let me look at what the conversion actually wrote:Now `whisper.py`. A local model folder is checked before loading, so these problems produce a clear error message instead of a cryptic one:Both checks behave correctly on your actual config files. The "not supported" line and "model.bin MISSING" come only from my test setup here. Saving:I've fixed the Chillarmo turbo model folder problem, but BuzzASR can't run in faster-whisper at all.

**Turbo-armenian: one command to fix.** The model converted correctly. Only `preprocessor_config.json` was missing, because newer `transformers` saves that file under a different name. The conversion script now writes it, and `--fix` repairs a folder that's already converted:

```
python -m stt.convert_model --fix models/whisper-turbo-hy
python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy --compute-types int8_float32 --beams 1 5
```

It should take about 0.9 s per phrase, like stock turbo. The real question is whether its Armenian is better.

**BuzzASR: I suggest dropping it.** Its custom tokenizer puts Whisper's special tokens at ids 0–7. faster-whisper assumes the standard layout, so the output would be empty or garbled. Getting around that means either rewriting faster-whisper's tokenizer handling, or running it in plain `transformers`. Plain `transformers` would be too slow on a GTX 1650 in float16, and at about 6 GB in float32 it won't fit in 4 GB. I'd mark it "❌ Rejected: incompatible with faster-whisper on this hardware" in DECISION.md. That's a real finding for the README. `whisper.py` now refuses such folders with that explanation instead of the misleading "English-only" error.

**Files keep getting reverted.** `stt/whisper.py` on your laptop was set back to an older version again: beam 5 was back, timestamped after my save. Earlier the same thing happened to `bench/stt_speed.py`. I've saved both again, checking first that nothing newer would be overwritten. Is something else writing to this folder, like another Claude session, a git command, or an editor with old tabs open? Until we find it, my changes may quietly disappear between runs.

I still need what you actually said in `mic.wav` to judge the accuracy of the two models.

**PROMPT**

```
7a(.venv) PS C:\Chris\livetranslation> python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy --quantization int8_float32
W1022 16:46:29.475000 17664 .venv\Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
Converting Chillarmo/whisper-large-v3-turbo-armenian -> models\whisper-turbo-hy (int8_float32) ...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.20k/1.20k [00:00<00:00, 1.15MB/s]
model.safetensors: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.04GB, 26.3MB/s  
model.safetensors: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 3.24GB / 3.24GB, 60.4MB/s  
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 587/587 [00:02<00:00, 286.91it/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.72k/3.72k [00:00<00:00, 5.34MB/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 283k/283k [00:00<00:00, 387MB/s]
vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 4.88MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.71M/2.71M [00:00<00:00, 27.5MB/s]
merges.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 494k/494k [00:00<00:00, 44.6MB/s]
normalizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.7k/52.7k [00:00<00:00, 38.9MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 34.6k/34.6k [00:00<00:00, 21.6MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.19k/2.19k [00:00<00:00, 2.47MB/s]
Saving tokenizer + preprocessor config ...
preprocessor_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [00:00<00:00, 668kB/s]
  model.bin                    ok
  config.json                  ok
  tokenizer.json               ok
  preprocessor_config.json     MISSING
Done.
(.venv) PS C:\Chris\livetranslation> python -m stt.convert_model BuzzASR/armenian models/buzzasr-hy --quantization int8_float32
W1022 16:49:35.406000 2624 .venv\Lib\site-packages\torch\distributed\elastic\multiprocessing\redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
Converting BuzzASR/armenian -> models\buzzasr-hy (int8_float32) ...
config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.20k/1.20k [00:00<00:00, 1.15MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.07GB, 24.2MB/s  
model.safetensors: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 3.09GB / 3.09GB, 61.4MB/s  
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1259/1259 [00:01<00:00, 650.60it/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 471/471 [00:00<00:00, 2.82MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 283k/283k [00:00<00:00, 62.8MB/s]
vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.87M/1.87M [00:00<00:00, 12.8MB/s]
merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.29M/1.29M [00:00<00:00, 54.4MB/s]
normalizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52.7k/52.7k [00:00<00:00, 12.5MB/s]
added_tokens.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 34.6k/34.6k [00:00<00:00, 21.5MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.19k/2.19k [00:00<00:00, 2.47MB/s]
Saving tokenizer + preprocessor config ...
preprocessor_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 357/357 [00:00<00:00, 630kB/s]
  model.bin                    ok
  config.json                  ok
  tokenizer.json               ok
  preprocessor_config.json     MISSING
Done.
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy models/buzzasr-hy --compute-types int8_float32 --beams 1 5 
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 5.68, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [int8_float32] ===
  load+warm-up 11.8 s, VRAM for this model ~2077 MiB
  beam 1 [ 1.86- 3.78s]  1.28s lp=-0.42 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  1.13s lp=-0.25 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s]  1.54s lp=-0.42 gain=2.18  Հավովով բաշէվ սզս։
  beam 5 [ 5.41- 6.34s]  1.25s lp=-0.22 gain=6.07  Չպես սեք։

=== models/whisper-turbo-hy  [int8_float32] ===
  load failed: ValueError: Invalid input features shape: expected an input with shape (1, 128, 3000), but got an input with shape (1, 80, 3000) instead

=== models/buzzasr-hy  [int8_float32] ===
  load failed: ValueError: 'models/buzzasr-hy' is English-only; an Armenian-capable model is needed

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 int8_float32      1    1.21    1.28  OK
large-v3                                 int8_float32      5    1.39    1.54  TOO SLOW
```

**RESPONSE**

This log matches your previous run: same timings, and the old error messages ("English-only"). The conversion output also comes from the old script, which didn't have the mel-bin or tokenizer checks yet. So this looks like the run from before my fix. Let me check what's on your laptop now:Your laptop now has the latest versions of `whisper.py`, `convert_model.py` and `stt_speed.py`. They haven't been reverted since my last save.

The run you pasted is from before my fix: the models were converted at 16:46–16:49 with the old conversion script. The new one would also have printed the mel bins and a tokenizer check. `models/whisper-turbo-hy` still has no `preprocessor_config.json`, which is why turbo fails with the 80-vs-128 error.

Nothing needs downloading or converting again. Just repair the folder and rerun the benchmark:

```
python -m stt.convert_model --fix models/whisper-turbo-hy
python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy --compute-types int8_float32 --beams 1 5
```

The first command should end with `mel bins: 128` and `tokenizer: ok`. I've left BuzzASR out of the benchmark: its tokenizer stores Whisper's special tokens in a non-standard order, which faster-whisper can't decode. You can confirm with `python -m stt.convert_model --fix models/buzzasr-hy`, which will report it as not usable and explain why.

**PROMPT**

```
(.venv) PS C:\Chris\livetranslation> python -m bench.stt_speed recordings/mic.wav --models large-v3 models/whisper-turbo-hy --compute-types int8_float32 --beams 1 5
=== Environment ===
python 3.13.12
torch 2.11.0+cu128 (CUDA build 12.8), cuda available: True
  GPU 0: NVIDIA GeForce GTX 1650, dedicated VRAM 4.0 GiB
ctranslate2 4.8.3, CUDA devices: 1
  compute types: ['float16', 'float32', 'int8', 'int8_float16', 'int8_float32']
nvidia-smi: NVIDIA GeForce GTX 1650, 4096, 0, P8, 5.54, [N/A], 621.01
  (name, total MiB, used MiB, perf state, power W, power limit W, driver)
  Very low power limit or P8 while busy = power saving: plug in the charger and
  set the Windows power mode to 'Best performance'.

recordings/mic.wav: 2 phrase(s)

=== large-v3  [int8_float32] ===
  load+warm-up 11.3 s, VRAM for this model ~2077 MiB
  beam 1 [ 1.86- 3.78s]  1.28s lp=-0.42 gain=2.18  Հավով բաշէվ սզս։
  beam 1 [ 5.41- 6.34s]  1.13s lp=-0.25 gain=6.07  Չպես սյկ։
  beam 5 [ 1.86- 3.78s]  1.54s lp=-0.42 gain=2.18  Հավովով բաշէվ սզս։
  beam 5 [ 5.41- 6.34s]  1.25s lp=-0.22 gain=6.07  Չպես սեք։

=== models/whisper-turbo-hy  [int8_float32] ===
  load+warm-up 2.7 s, VRAM for this model ~1149 MiB
  beam 1 [ 1.86- 3.78s]  0.88s lp=-0.33 gain=2.18  - Ա՜, պաշև, ձեզ…
  beam 1 [ 5.41- 6.34s]  0.86s lp=-0.07 gain=6.07  - Չպես եք…
  beam 5 [ 1.86- 3.78s]  0.91s lp=-0.31 gain=2.18  Ավո, պաշև ձեզ…
  beam 5 [ 5.41- 6.34s]  0.88s lp=-0.07 gain=6.07  - Չպես եք…

=== Summary (budget: STT <= 1.5 s per phrase) ===
model                                    compute        beam  mean s   max s
large-v3                                 int8_float32      1    1.21    1.28  OK
large-v3                                 int8_float32      5    1.40    1.54  TOO SLOW
models/whisper-turbo-hy                  int8_float32      1    0.87    0.88  OK
models/whisper-turbo-hy                  int8_float32      5    0.89    0.91  OK
```

**RESPONSE**

The Armenian fine-tune works and decodes a phrase in about 0.9 s, the fastest result so far. Its output also carries subtitle-style marks: a leading "- " and a trailing "…". I'll strip those and make this model the default. First, a check that the laptop copy hasn't changed:The Armenian fine-tune (Chillarmo's turbo-armenian, now in `models/whisper-turbo-hy`) beats stock large-v3 on both speed and accuracy. I've made it the default in `stt/whisper.py`.

| | large-v3 (int8) | turbo-hy (int8) |
|---|---|---|
| Time per phrase | 1.2 s (beam 1), 1.5 s (beam 5) | **0.9 s** at beam 1 or 5 |
| VRAM | 2.1 GB | **1.1 GB** |
| Confidence (closer to 0 is better) | −0.42 / −0.25 | **−0.31 / −0.07** |
| Phrase 1 | Հավով բաշէվ սզս | Ավո, պաշև ձեզ |
| Phrase 2 | Չպես սեք | Չպես եք |

I'm guessing you said "Ալո, բարև ձեզ" and "Ինչպե՞ս եք". If so, turbo-hy is nearly right. "Չպես եք" is how "Ինչպե՞ս եք" sounds in casual speech, and "պաշև" for "բարև" is a sound mix-up the quiet mic makes more likely. Is that what you said?

Changes in `stt/whisper.py`:

- **Default model** is now `models/whisper-turbo-hy` in int8_float32, at beam 5. On this model beam 5 costs only about 0.03 s more than beam 1.
- **Subtitle marks removed.** The model was trained on subtitle-style text, so it adds a leading "- " and a trailing "…". Both are now stripped before translation.

This puts speech-to-text at about 0.9 s of the 5 s budget, and leaves about 2.9 GB of VRAM for translation.

Two phrases aren't enough to choose a model, so record the 10–15 test clips from PLAN.md and re-run the benchmark on them before treating this as final. Then I'd update DECISION.md:

- **Stage 3:** turbo-hy chosen, BuzzASR rejected (its tokenizer can't be decoded by faster-whisper).
- **VRAM budget:** 4 GB, not the planned 11.9 GB.

After that, the next step is translation (`translate/`).

**PROMPT**

```
does you changed whisper.py?
```

**RESPONSE**

Yes, I changed it in my last step, but your laptop doesn't have that version anymore. I just checked: `stt/whisper.py` on your laptop is back to the previous version (`large-v3`, beam 1, no dash/ellipsis cleanup). It was overwritten after I saved, so either you reverted it or something on your side did. Your editor would do this if it has the file open and saved its older copy, as would a `git checkout`/`restore`, or another session working in this folder.

My last step changed three things, all in `stt/whisper.py`:

1. **Default model:** `"large-v3"` → `"models/whisper-turbo-hy"`, the Armenian fine-tune that was faster and more accurate in your benchmark.
2. **Beam size:** 1 → 5. On turbo-hy this costs only about 0.03 s.
3. **Cleanup of subtitle marks:** a leading "- " and a trailing "…" or "..." are now stripped, so "- Չպես եք…" becomes "Չպես եք".

Everything else in the file is the same as the version you have now.

I haven't saved it again, in case you reverted it on purpose. Do you want these changes back on your laptop? If you used an editor, close or reload `whisper.py` there first so it doesn't overwrite them again.

**PROMPT**

```
delete unnecessary files that you created for STT testing
```

**RESPONSE**

I can't delete files on your laptop from this session, so I've cleaned up my side and listed below what to delete yourself. My scratch test copies, the stand-in test scripts, the extra copy of the benchmark and the cached Python files are all gone.

**Safe to delete in File Explorer** (all in `C:\Chris\livetranslation`):

- **`bench\` (the whole folder):** contains `stt_speed.py`, `__init__.py` and `__pycache__`. It was only used to compare models and settings on your GPU.
- **`models\buzzasr-hy\`:** about 1.5 GB. It can't run in faster-whisper, so it's of no use to the project.

**Keep these:**

- **`stt\__init__.py` and `stt\whisper.py`:** the speech-recognition module itself.
- **`stt\convert_model.py`:** `whisper.py` uses it to check a model folder before loading, and you'll need it to rebuild the model on another machine.
- **`models\whisper-turbo-hy\`:** the Armenian model you're using. Run `python -m stt.convert_model --fix models\whisper-turbo-hy` on it once, or it fails to load.

PLAN.md lists STT benchmarking (`bench/stt_bench.py`) as part of the test plan. If you want benchmark numbers in the README, keep `bench\stt_speed.py` instead of deleting it.

**PROMPT**

```
how can I test STT
```

**RESPONSE**

You can test speech-to-text on your laptop in three ways, from quickest to most thorough.

**1. Transcribe a recording** (checks that it runs and how good the text is)

```
python -m stt.convert_model --fix models\whisper-turbo-hy
python -m stt.whisper recordings\mic.wav --segment --model models\whisper-turbo-hy --beam-size 5
```

You only need the first command once: it adds the missing config file to the model folder. The second cuts the file into phrases the same way live mode does, then prints one line per phrase: time span, decode time, confidence (`lp`, closer to 0 is better) and the Armenian text. Phrases the filter drops are shown with the reason.

Pass `--model` and `--beam-size` explicitly: your `whisper.py` was reverted to the version that still defaults to `large-v3` at beam 1.

**2. Record new test clips**

Your current clip is very quiet, so first raise the mic level in Windows sound settings. Then record:

```
python -m audio.capture --list
python -m audio.capture --record 10 --out recordings/
```

Speak a few known sentences, then run step 1 on the new file. Also test the cases that cause hallucinations:

- **Silence:** a clip with nobody speaking should print nothing.
- **Background noise or music:** should be dropped, not turned into made-up text.
- **Long phrases:** over 8 s, which the segmenter force-cuts.
- **The call side:** play an Armenian video or call and record `system.wav`.

**3. Measure accuracy (what PLAN.md §6 asks for)**

Record the 10–15 clips from the plan and write down exactly what was said in each, as a `.txt` file next to each `.wav`. A script can then compute word and character error rates (WER and CER) with `jiwer`, which is already in `requirements.txt`, plus decode time per phrase. That gives you numbers for the README instead of judging by eye. I haven't written that script yet. Want me to?

**PROMPT**

```
Can I run stt/whisper.py?
```

**RESPONSE**

Yes, but run it as a module from the project folder, not as a file path:

```
cd C:\Chris\livetranslation
python -m stt.whisper recordings\mic.wav --segment --model models\whisper-turbo-hy --beam-size 5
```

`python stt\whisper.py ...` fails with `ModuleNotFoundError: No module named 'core'`. `whisper.py` imports `core.types` and `audio.*`, and Python only finds those when it starts from the project folder, which `-m` does.

Options:

- `recordings\mic.wav`: any WAV or FLAC file. It's converted to 16 kHz mono automatically.
- `--segment`: cuts the file into phrases the way live mode does. Without it, the whole file is decoded as one piece.
- `--model`: a model name like `large-v3`, or a converted model folder.
- `--beam-size`: 5 for `whisper-turbo-hy`, 1 for `large-v3`.
- `--device` and `--compute-type`: the defaults, `cuda` and `int8_float32`, are already the best settings for your GTX 1650.
- `-v`: also prints model load and warm-up times.

If you haven't yet, run `python -m stt.convert_model --fix models\whisper-turbo-hy` once first. Otherwise loading that model fails on a missing config file.