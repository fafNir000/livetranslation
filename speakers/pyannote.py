"""Speaker identification with pyannote embeddings + online clustering.

Why not the full pyannote diarization pipeline? It processes a finished
recording all at once, and its labels can change on every rerun. We need a
label per segment, *now*, that never changes afterwards. So we:

  1. compute one voice embedding per system-audio segment with pyannote's
     WeSpeaker ResNet34 model (fast: a few ms on GPU per segment), then
  2. compare it to the running average ("centroid") of each known speaker:
       - cosine similarity >= match_threshold  -> same speaker, update centroid
       - otherwise, if the segment is long enough -> new "Speaker N"
       - too short to trust                     -> closest known speaker

Microphone segments are always "Me" and are never embedded.

Run as a module (NOT `python speakers/pyannote.py`: running the file directly
puts speakers/ on sys.path and this file would shadow the real `pyannote`
package). To tune the threshold on your own clips:

    python -m speakers.pyannote clip1.wav clip2.wav clip3.wav

which prints the pairwise similarity matrix and the label each clip gets.
Clips named me_*.wav go through the microphone path and are labelled "Me";
all others go through system-audio clustering (Speaker 1, Speaker 2, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from core.types import SAMPLE_RATE, Segment
from speakers import ME

DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


@dataclass
class SpeakerParams:
    match_threshold: float = 0.5  # cosine sim to join an existing speaker. TUNE on real clips
    min_embed_s: float = 0.6  # shorter segments aren't embedded -> reuse last speaker
    min_new_speaker_s: float = 1.2  # only segments this long may create a new speaker
    max_speakers: int = 8  # safety cap against runaway over-splitting
    centroid_max_weight_s: float = 30.0  # caps centroid inertia so it can still adapt a bit


@dataclass
class _Speaker:
    number: int
    centroid: np.ndarray  # L2-normalised
    weight_s: float  # seconds of speech merged into the centroid
    segments: int = 1


@dataclass
class OnlineSpeakerClustering:
    """Pure-numpy online clustering (no torch), so it can be unit-tested."""

    params: SpeakerParams = field(default_factory=SpeakerParams)
    speakers: list[_Speaker] = field(default_factory=list)

    def assign(self, embedding: np.ndarray, duration_s: float) -> tuple[int, float]:
        """Return (speaker number starting at 1, best cosine similarity)."""
        e = _normalise(embedding)
        p = self.params

        if not self.speakers:
            return self._new(e, duration_s), 1.0

        sims = np.array([s.centroid @ e for s in self.speakers])
        best = int(np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim >= p.match_threshold:
            self._update(self.speakers[best], e, duration_s)
            return self.speakers[best].number, best_sim

        if duration_s >= p.min_new_speaker_s and len(self.speakers) < p.max_speakers:
            return self._new(e, duration_s), best_sim

        # Unsure: short segment or too many speakers. Choose the closest, but don't
        # let an uncertain match pull that speaker's centroid.
        return self.speakers[best].number, best_sim

    def _new(self, e: np.ndarray, duration_s: float) -> int:
        spk = _Speaker(number=len(self.speakers) + 1, centroid=e, weight_s=duration_s)
        self.speakers.append(spk)
        return spk.number

    def _update(self, spk: _Speaker, e: np.ndarray, duration_s: float) -> None:
        w = min(spk.weight_s, self.params.centroid_max_weight_s)
        spk.centroid = _normalise(spk.centroid * w + e * duration_s)
        spk.weight_s += duration_s
        spk.segments += 1


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


class PyannoteSpeakerID:
    """`identify(segment) -> label`. Call from ONE worker thread only."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        params: SpeakerParams | None = None,
        hf_token: str | None = None,
    ) -> None:
        import torch
        from pyannote.audio import Inference, Model

        self._torch = torch
        token = hf_token or os.environ.get("HF_TOKEN")
        model = _load_model(Model, model_name, token)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.inference = Inference(model, window="whole")
        self.inference.to(self.device)

        self.clustering = OnlineSpeakerClustering(params or SpeakerParams())
        self._last_system_label: str | None = None
        self.last_similarity: float | None = None  # handy for --debug output

    # ------------------------------------------------------------------ API
    def identify(self, segment: Segment) -> str:
        if segment.source == "mic":
            return ME

        p = self.clustering.params
        if segment.duration < p.min_embed_s:
            # Too short for a reliable embedding: assume the last speaker went on.
            self.last_similarity = None
            return self._last_system_label or self._label(
                self.clustering.assign(self.embed(segment.pcm), segment.duration)[0]
            )

        number, sim = self.clustering.assign(self.embed(segment.pcm), segment.duration)
        self.last_similarity = sim
        self._last_system_label = self._label(number)
        return self._last_system_label

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        waveform = self._torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))[None]
        emb = self.inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return _normalise(emb)

    def warmup(self) -> None:
        """Run once at startup so the first real segment isn't slow (CUDA init)."""
        self.embed(np.zeros(SAMPLE_RATE, dtype=np.float32))

    @staticmethod
    def _label(number: int) -> str:
        return f"Speaker {number}"


def _load_model(Model, name: str, token: str | None):
    """pyannote 4.x takes `token=`, 3.x takes `use_auth_token=`."""
    try:
        model = Model.from_pretrained(name, token=token)
    except TypeError:
        model = Model.from_pretrained(name, use_auth_token=token)
    if model is None:  # pyannote returns None instead of raising on auth problems
        raise RuntimeError(
            f"Could not load '{name}'. If the model is gated, accept its terms on "
            "huggingface.co and put HF_TOKEN=... in your .env file."
        )
    return model


# --------------------------------------------------------------------------- #
# Threshold tuning helper
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    import argparse
    import time

    from audio.file_source import load_audio

    ap = argparse.ArgumentParser(description="Embed clips and show speaker similarities")
    ap.add_argument("wavs", nargs="+", help="one clip per segment, e.g. a.wav b.wav")
    ap.add_argument("--threshold", type=float, default=SpeakerParams.match_threshold)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--mic-prefix", default="me",
        help="clips whose file name starts with this are treated as microphone audio "
        "(labelled Me, like in the real pipeline). Default: me",
    )
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    sid = PyannoteSpeakerID(device=args.device, params=SpeakerParams(match_threshold=args.threshold))
    sid.warmup()
    print(f"Model on {sid.device}\n")

    names, embs = [], []
    for path in args.wavs:
        pcm = load_audio(path)
        t = time.perf_counter()
        embs.append(sid.embed(pcm))
        ms = (time.perf_counter() - t) * 1000
        name = os.path.basename(path)
        names.append(name)
        # Same routing as the live pipeline: mic -> "Me", system -> clustering.
        source = "mic" if name.lower().startswith(args.mic_prefix.lower()) else "system"
        seg = Segment(source, 0.0, len(pcm) / SAMPLE_RATE, pcm)
        label = sid.identify(seg)
        print(f"{name:<30} {source:<6} {seg.duration:5.1f}s  {ms:5.0f} ms  -> {label}")

    sims = np.stack(embs) @ np.stack(embs).T
    w = max(len(n) for n in names)
    print(
        "\nCosine similarity of ALL clips, mic ones included, to check that the voices differ"
        "\n(same person usually higher than different people):"
    )
    print(" " * (w + 2) + "  ".join(f"{i:>5}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"{n:<{w}}  " + "  ".join(f"{s:5.2f}" for s in sims[i]))


if __name__ == "__main__":
    main()
