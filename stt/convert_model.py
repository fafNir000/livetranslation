"""Convert a Hugging Face Whisper fine-tune to CTranslate2 so faster-whisper can run it.

    python -m stt.convert_model Chillarmo/whisper-large-v3-turbo-armenian models/whisper-turbo-hy
    python -m stt.convert_model --fix models/whisper-turbo-hy      # repair an existing folder

Then use the output folder as the model name:
    python -m stt.whisper recordings/mic.wav --segment --model models/whisper-turbo-hy

Why a script instead of the plain `ct2-transformers-converter` command:
faster-whisper needs `tokenizer.json` and `preprocessor_config.json` NEXT TO the
converted weights. Without the tokenizer it silently falls back to the stock
OpenAI tokenizer; without the preprocessor config it assumes 80 mel bins, but
large-v3-based models need 128 ("expected (1, 128, 3000), got (1, 80, 3000)").
Recent transformers versions save the processor as ONE `processor_config.json`
(feature extractor nested inside), which faster-whisper doesn't read, so we
write the flat `preprocessor_config.json` ourselves.

It also checks the tokenizer layout. faster-whisper assumes the standard Whisper
vocabulary (text tokens first, then <|endoftext|> = 50257, then special tokens).
A fine-tune with its own tokenizer that moves the special tokens (BuzzASR/armenian
puts <|endoftext|> at id 0) cannot be decoded by faster-whisper and is rejected.

Needs `transformers` and `torch` (already in requirements.txt). The models/
folder holds weights: keep it in .gitignore.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def ensure_preprocessor_config(model_dir: str | Path) -> bool:
    """Write the flat preprocessor_config.json faster-whisper reads. Returns True if written."""
    d = Path(model_dir)
    target = d / "preprocessor_config.json"
    if target.exists():
        return False
    combined = d / "processor_config.json"
    if not combined.exists():
        raise FileNotFoundError(f"{d}: neither preprocessor_config.json nor processor_config.json")
    cfg = json.loads(combined.read_text(encoding="utf-8"))
    fe = cfg.get("feature_extractor", cfg)
    if "feature_size" not in fe:
        raise ValueError(f"{combined}: no feature_size found")
    target.write_text(json.dumps(fe, indent=2), encoding="utf-8")
    return True


def check_tokenizer_layout(model_dir: str | Path) -> str:
    """Return "" if faster-whisper can decode this vocabulary, else the problem."""
    tok = Path(model_dir) / "tokenizer.json"
    if not tok.exists():
        return "tokenizer.json missing (faster-whisper would fall back to the stock tokenizer)"
    added = {t["content"]: t["id"] for t in json.loads(tok.read_text(encoding="utf-8"))["added_tokens"]}
    eot, nots = added.get("<|endoftext|>"), added.get("<|notimestamps|>")
    if eot is None or nots is None or "<|hy|>" not in added:
        return "special tokens <|endoftext|> / <|notimestamps|> / <|hy|> not found"
    if eot < 50257 or nots < eot:
        return (f"custom vocabulary layout (<|endoftext|>={eot}, <|notimestamps|>={nots}); "
                "faster-whisper treats every token >= <|endoftext|> as special and every token "
                "> <|notimestamps|> as a timestamp, so its output would be empty or garbage")
    return ""


def report(model_dir: str | Path) -> None:
    d = Path(model_dir)
    for f in ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"):
        print(f"  {f:28} {'ok' if (d / f).exists() else 'MISSING'}")
    if (d / "preprocessor_config.json").exists():
        n_mels = json.loads((d / "preprocessor_config.json").read_text(encoding="utf-8"))["feature_size"]
        print(f"  mel bins: {n_mels}")
    problem = check_tokenizer_layout(d)
    print(f"  tokenizer: {'ok' if not problem else 'NOT USABLE - ' + problem}")


def convert(model_id: str, out_dir: str, quantization: str = "int8_float32", force: bool = False) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from ctranslate2.converters import TransformersConverter
    from transformers import AutoProcessor

    out = Path(out_dir)
    print(f"Converting {model_id} -> {out} ({quantization}) ...")
    TransformersConverter(model_id, load_as_float16=quantization.startswith("float16")).convert(
        str(out), quantization=quantization, force=force
    )
    print("Saving tokenizer + preprocessor config ...")
    AutoProcessor.from_pretrained(model_id).save_pretrained(str(out))
    ensure_preprocessor_config(out)
    report(out)
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_id", nargs="?", help="Hugging Face repo id or local HF model folder")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--fix", metavar="DIR", help="only repair/check an already converted folder")
    ap.add_argument("--quantization", default="int8_float32",
                    help="int8_float32 (best on GTX 16xx) | float16 | int8_float16 | int8")
    ap.add_argument("--force", action="store_true", help="overwrite out_dir")
    args = ap.parse_args()

    if args.fix:
        written = ensure_preprocessor_config(args.fix)
        print(f"{args.fix}: preprocessor_config.json {'written' if written else 'already present'}")
        report(args.fix)
        return
    if not (args.model_id and args.out_dir):
        ap.error("give MODEL_ID OUT_DIR, or --fix DIR")
    convert(args.model_id, args.out_dir, args.quantization, args.force)


if __name__ == "__main__":
    main()
