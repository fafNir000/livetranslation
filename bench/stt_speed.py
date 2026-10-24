"""Which Whisper setup fits the latency budget on THIS machine?

1. Prints what the GPU really is: name, dedicated VRAM, VRAM already in use,
   power state, and what torch / CTranslate2 can see. (The Windows "11.9 GB"
   figure is often dedicated VRAM + shared system RAM; a model that spills into
   shared RAM runs ~10x slower.)
2. Cuts the clip into phrases (same Silero segmenter as live mode) and, for every
   model x compute type x beam size, prints decode time per phrase, VRAM used,
   whether temperature fallback re-decoded it, and the Armenian text, so speed
   AND accuracy can be compared side by side.

    python -m bench.stt_speed recordings/mic.wav
    python -m bench.stt_speed recordings/mic.wav --models large-v3-turbo models/whisper-turbo-hy
    python -m bench.stt_speed recordings/mic.wav --compute-types float16 int8_float16 --beams 1 5

Budget (PLAN.md §4): STT <= 1.5 s per phrase.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

STT_BUDGET_S = 1.5


def nvidia_smi(query: str) -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found"
    try:
        return subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def vram_used_mib() -> int | None:
    try:
        return int(nvidia_smi("memory.used").splitlines()[0])
    except (ValueError, IndexError):
        return None


def report_environment() -> None:
    print("=== Environment ===")
    print("python", sys.version.split()[0])
    try:
        import torch

        print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), "
              f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU 0: {p.name}, dedicated VRAM {p.total_memory / 2**30:.1f} GiB")
    except ImportError:
        print("torch not installed")
    import ctranslate2

    n = ctranslate2.get_cuda_device_count()
    print(f"ctranslate2 {ctranslate2.__version__}, CUDA devices: {n}")
    if n:
        print(f"  compute types: {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
    print("nvidia-smi:",
          nvidia_smi("name,memory.total,memory.used,pstate,power.draw,power.limit,driver_version"))
    print("  (name, total MiB, used MiB, perf state, power W, power limit W, driver)")
    print("  Very low power limit or P8 while busy = power saving: plug in the charger and")
    print("  set the Windows power mode to 'Best performance'.\n")


def main() -> None:
    from audio.file_source import load_audio
    from core.types import SAMPLE_RATE
    from stt.whisper import WhisperConfig, WhisperSTT, offline_segments

    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--models", nargs="+", default=["large-v3", "large-v3-turbo"],
                    help="model names or local CTranslate2 folders")
    ap.add_argument("--compute-types", nargs="+", default=["int8_float32", "int8_float16"])
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-segment", action="store_true", help="decode the whole file as one piece")
    args = ap.parse_args()

    report_environment()

    audio = load_audio(args.file)
    if args.no_segment:
        phrases = [(0.0, len(audio) / SAMPLE_RATE, audio)]
    else:
        phrases = [(s.t_start, s.t_end, s.pcm) for s in offline_segments(audio)]
    print(f"{args.file}: {len(phrases)} phrase(s)\n")

    vram0 = vram_used_mib()
    summary = []
    for model in args.models:
        for ct in args.compute_types:
            print(f"=== {model}  [{ct}] ===")
            t0 = time.perf_counter()
            try:
                stt = WhisperSTT(WhisperConfig(model=model, device=args.device, compute_type=ct))
            except Exception as e:  # OOM, missing files, unsupported type...
                print(f"  load failed: {type(e).__name__}: {e}\n")
                continue
            load_s = time.perf_counter() - t0
            vram = vram_used_mib()
            vram_txt = f"~{vram - vram0} MiB" if vram is not None and vram0 is not None else "?"
            print(f"  load+warm-up {load_s:.1f} s, VRAM for this model {vram_txt}")
            try:
                total = int(nvidia_smi("memory.total").splitlines()[0])
                if vram is not None and vram > 0.85 * total:
                    print(f"  WARNING: {vram}/{total} MiB VRAM in use - Windows will page to "
                          "system RAM and everything gets ~10x slower")
            except (ValueError, IndexError):
                pass

            for beam in args.beams:
                stt.cfg.beam_size = beam
                times = []
                for t_s, t_e, pcm in phrases:
                    r = stt.transcribe(pcm)
                    times.append(r.elapsed_s)
                    temps = r.extra.get("temperatures", [])
                    fb = " [FALLBACK]" if any(t > 0 for t in temps) else ""
                    txt = r.text if r.ok else f"-- dropped: {r.reason} (raw: {r.raw_text!r})"
                    print(f"  beam {beam} [{t_s:5.2f}-{t_e:5.2f}s] {r.elapsed_s:5.2f}s "
                          f"lp={r.avg_logprob:5.2f} gain={r.extra.get('gain', 1)}{fb}  {txt}")
                if times:
                    worst = max(times)
                    summary.append((model, ct, beam, float(np.mean(times)), worst,
                                    "OK" if worst <= STT_BUDGET_S else "TOO SLOW"))
            del stt  # free VRAM before the next config
            gc.collect()
            print()

    print(f"=== Summary (budget: STT <= {STT_BUDGET_S} s per phrase) ===")
    print(f"{'model':40} {'compute':14} {'beam':>4} {'mean s':>7} {'max s':>7}")
    for m, ct, b, mean, worst, v in summary:
        print(f"{m:40} {ct:14} {b:>4} {mean:7.2f} {worst:7.2f}  {v}")


if __name__ == "__main__":
    main()
