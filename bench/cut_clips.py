"""Cut a recording into fixed-length clips for speaker-ID threshold tuning.

    python cut_clips.py recordings\\mic.wav me
    python cut_clips.py recordings\\system.wav other --seconds 3 --out clips

Clips that are mostly silence are skipped (they make similarity scores meaningless).
Output: clips\\me_1.wav, clips\\me_2.wav, ...
"""

import argparse
import os

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut a WAV into fixed-length clips")
    ap.add_argument("path", help="input audio file")
    ap.add_argument("prefix", help="name prefix for the clips, e.g. me / other")
    ap.add_argument("--seconds", type=float, default=3.0, help="clip length (default 3)")
    ap.add_argument("--out", default="clips", help="output folder (default clips)")
    ap.add_argument(
        "--min-db", type=float, default=-45.0,
        help="skip clips quieter than this RMS level in dBFS (default -45)",
    )
    args = ap.parse_args()

    audio, sr = sf.read(args.path, dtype="float32")
    n = int(args.seconds * sr)
    if len(audio) < n:
        raise SystemExit(f"{args.path} is shorter than {args.seconds} s")

    os.makedirs(args.out, exist_ok=True)
    written = skipped = 0
    for i in range(0, len(audio) - n + 1, n):
        clip = audio[i : i + n]
        rms_db = 20 * np.log10(np.sqrt(np.mean(clip**2)) + 1e-9)
        if rms_db < args.min_db:
            skipped += 1
            continue
        written += 1
        out = os.path.join(args.out, f"{args.prefix}_{written}.wav")
        sf.write(out, clip, sr)
        print(f"{out}  ({rms_db:.1f} dBFS)")

    print(f"\n{written} clips written, {skipped} silent clips skipped")


if __name__ == "__main__":
    main()
