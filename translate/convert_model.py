"""Convert an NLLB-200 checkpoint from Hugging Face to CTranslate2 (run once, needs network).

    python -m translate.convert_model                                   # default: 1.3B distilled
    python -m translate.convert_model facebook/nllb-200-distilled-600M models/nllb-200-distilled-600M
    python -m translate.convert_model facebook/nllb-200-3.3B models/nllb-200-3.3B --quantization int8_float32
    python -m translate.convert_model --check models/nllb-200-distilled-1.3B

Then point the translator at the output folder:
    python -m translate.nllb --model models/nllb-200-distilled-1.3B "Բարև, ինչպե՞ս ես։"

Why a script instead of the plain `ct2-transformers-converter` command: it also
saves the tokenizer (sentencepiece model + language codes) INTO the model
folder, so at runtime nothing is fetched from the Hugging Face Hub and the
program works offline in the interview. It then checks that the folder really
has the Armenian and English language tokens.

Size on disk / VRAM (approx.): 600M ~0.6 GB int8, 1.3B ~1.4 GB int8 / 2.7 GB
float16, 3.3B ~3.4 GB int8. The conversion itself loads the full model in RAM
(1.3B: ~3 GB as float16).

License note: NLLB-200 weights are CC-BY-NC-4.0 (fine for this challenge, not
for commercial use). The models/ folder holds weights: keep it in .gitignore.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
DEFAULT_OUT_DIR = "models/nllb-200-distilled-1.3B"
_TOKENIZER_FILES = ("sentencepiece.bpe.model", "tokenizer.json")


def _vocabulary(model_dir: Path) -> set[str] | None:
    """Target vocabulary tokens of a CTranslate2 model folder (None if not found)."""
    for name in ("shared_vocabulary.json", "target_vocabulary.json"):
        p = model_dir / name
        if p.exists():
            return set(json.loads(p.read_text(encoding="utf-8")))
    for name in ("shared_vocabulary.txt", "target_vocabulary.txt"):
        p = model_dir / name
        if p.exists():
            return {ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines()}
    return None


def check_model_dir(
    model_dir: str | Path,
    src_lang: str = "hye_Armn",
    tgt_lang: str = "eng_Latn",
    need_tokenizer: bool = True,
) -> str:
    """Return "" if the folder is a usable CTranslate2 NLLB model, else the problem + fix."""
    d = Path(model_dir)
    fix = f"re-run: python -m translate.convert_model <hf model id> {d}"
    if not (d / "model.bin").exists():
        return f"model.bin missing (not a CTranslate2 model folder); {fix}"
    if need_tokenizer and not any((d / f).exists() for f in _TOKENIZER_FILES):
        return f"tokenizer files missing ({' / '.join(_TOKENIZER_FILES)}); {fix}"
    vocab = _vocabulary(d)
    if vocab is None:
        return f"no vocabulary file found; {fix}"
    missing = [t for t in (src_lang, tgt_lang) if t not in vocab]
    if missing:
        return f"language token(s) {missing} not in the vocabulary (is this an NLLB model?)"
    return ""


def report(model_dir: str | Path) -> None:
    d = Path(model_dir)
    for f in ("model.bin", "config.json", "shared_vocabulary.json", *_TOKENIZER_FILES):
        print(f"  {f:28} {'ok' if (d / f).exists() else 'missing'}")
    if (d / "model.bin").exists():
        print(f"  model.bin size: {(d / 'model.bin').stat().st_size / 1e9:.2f} GB")
    problem = check_model_dir(d)
    print(f"  check: {'ok' if not problem else 'NOT USABLE - ' + problem}")


def convert(model_id: str, out_dir: str, quantization: str = "int8_float32", force: bool = False) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from ctranslate2.converters import TransformersConverter
    from transformers import AutoTokenizer

    out = Path(out_dir)
    print(f"Converting {model_id} -> {out} ({quantization}) ... (downloads the model on first run)")
    TransformersConverter(
        model_id,
        load_as_float16=quantization != "float32",  # halves RAM during conversion
        low_cpu_mem_usage=True,
    ).convert(str(out), quantization=quantization, force=force)
    print("Saving tokenizer ...")
    AutoTokenizer.from_pretrained(model_id).save_pretrained(str(out))
    report(out)
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_id", nargs="?", default=DEFAULT_MODEL_ID,
                    help=f"Hugging Face repo id or local HF folder (default {DEFAULT_MODEL_ID})")
    ap.add_argument("out_dir", nargs="?", default=None,
                    help=f"output folder (default {DEFAULT_OUT_DIR} for the default model)")
    ap.add_argument("--check", metavar="DIR", help="only check an already converted folder")
    ap.add_argument("--quantization", default="int8_float32",
                    help="int8_float32 (best on GTX 16xx) | float16 | int8_float16 | int8")
    ap.add_argument("--force", action="store_true", help="overwrite out_dir")
    args = ap.parse_args()

    if args.check:
        report(args.check)
        return
    out_dir = args.out_dir or (
        DEFAULT_OUT_DIR if args.model_id == DEFAULT_MODEL_ID
        else os.path.join("models", args.model_id.rstrip("/").split("/")[-1])
    )
    convert(args.model_id, out_dir, args.quantization, args.force)


if __name__ == "__main__":
    main()
