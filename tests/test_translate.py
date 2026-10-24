"""Unit tests for translate/nllb.py (pure logic, no model or GPU needed).

The engine test uses a fake tokenizer + fake CTranslate2 translator, so it
checks the plumbing (sentence batching, language-token handling, scores,
flags) and runs anywhere:  pytest tests/test_translate.py
"""

from __future__ import annotations

from types import SimpleNamespace

from translate.nllb import (
    NllbConfig,
    NllbTranslator,
    armenian_ratio,
    check_output,
    clean_output,
    collapse_repeats,
    prepare_source,
    split_sentences,
)


# ------------------------------------------------------------- text helpers
def test_prepare_source_maps_ascii_full_stop_and_spaces():
    assert prepare_source("  Բարև   ձեզ:  Ինչպե՞ս եք: ") == "Բարև ձեզ։ Ինչպե՞ս եք։"
    # a colon after a Latin word / number is left alone
    assert prepare_source("ժամը 10:30 է") == "ժամը 10:30 է"


def test_split_sentences_merges_short_pieces():
    assert split_sentences("Այո։ Վաղը կգամ։") == ["Այո։ Վաղը կգամ։"]
    assert split_sentences("Բարև։ Ինչպե՞ս ես։ Ես լավ եմ։") == ["Բարև։ Ինչպե՞ս ես։", "Ես լավ եմ։"]
    assert split_sentences("Ես լավ եմ։ Լավ։") == ["Ես լավ եմ։ Լավ։"]
    assert split_sentences("Լավ") == ["Լավ"]


def test_split_sentences_ignores_in_word_question_mark():
    # "՞" sits inside the word, it is not a sentence end
    assert split_sentences("Ինչպե՞ս ես դու այսօր") == ["Ինչպե՞ս ես դու այսօր"]


def test_collapse_repeats():
    assert collapse_repeats("I don't know, I don't know, I don't know, I don't know") == "I don't know,"
    assert collapse_repeats("very, very good") == "very, very good"
    assert collapse_repeats("no no no no") == "no"


def test_clean_output_capitalizes():
    assert clean_output("  hello   there ") == "Hello there"


def test_armenian_ratio_and_flags():
    cfg = NllbConfig()
    assert armenian_ratio("Hello") == 0.0
    assert armenian_ratio("Բարև") == 1.0
    assert check_output("Hello, how are you?", -0.3, cfg) == ""
    assert "low confidence" in check_output("Hello", -3.0, cfg)
    assert "untranslated" in check_output("Բարև ձեզ friend", -0.3, cfg)


# ---------------------------------------------------------------- engine
class FakeTokenizer:
    """Whitespace 'tokenizer' with NLLB's layout: [src_lang] + tokens + [</s>]."""

    def __init__(self, src_lang="hye_Armn"):
        self.src_lang = src_lang
        self.vocab: dict[str, int] = {}

    def _id(self, t):
        return self.vocab.setdefault(t, len(self.vocab))

    def __call__(self, text, truncation=True, max_length=256):
        toks = [self.src_lang, *text.split(), "</s>"][:max_length]
        return {"input_ids": [self._id(t) for t in toks]}

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv[i] for i in ids]

    def convert_tokens_to_ids(self, toks):
        return [self._id(t) for t in toks]

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv[i] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in ("</s>", "eng_Latn", "hye_Armn")]
        return " ".join(toks)


class FakeTranslator:
    """Echoes a canned English sentence per source; records the call arguments."""

    def __init__(self, table, score=-0.4):
        self.table = table
        self.score = score
        self.calls = []

    # Only the keyword args the real ctranslate2.Translator.translate_batch accepts
    # (a subset), so passing an unsupported one fails here like it does on the GPU.
    _ALLOWED = {
        "target_prefix", "max_batch_size", "batch_type", "beam_size", "patience",
        "num_hypotheses", "length_penalty", "coverage_penalty", "repetition_penalty",
        "no_repeat_ngram_size", "disable_unk", "suppress_sequences", "end_token",
        "max_input_length", "max_decoding_length", "min_decoding_length", "return_scores",
        "sampling_topk", "sampling_topp", "sampling_temperature", "replace_unknowns",
    }

    def translate_batch(self, sources, **kw):
        bad = set(kw) - self._ALLOWED
        if bad:
            raise TypeError(f"translate_batch(): incompatible function arguments {sorted(bad)}")
        self.calls.append((sources, kw))
        out = []
        for src in sources:
            hy = " ".join(t for t in src if t not in ("hye_Armn", "</s>"))
            en = self.table.get(hy, "???")
            out.append(SimpleNamespace(hypotheses=[[kw["target_prefix"][0][0], *en.split()]],
                                       scores=[self.score]))
        return out


def _engine(table, score=-0.4, **cfg):
    tr = FakeTranslator(table, score)
    mt = NllbTranslator(NllbConfig(warmup=False, **cfg), translator=tr, tokenizer=FakeTokenizer())
    return mt, tr


def test_translate_single_sentence():
    mt, tr = _engine({"Բարև, ինչպե՞ս ես։": "hi, how are you?"})
    r = mt.translate("Բարև, ինչպե՞ս ես:", segment_id=7)
    assert r.ok and r.text == "Hi, how are you?"
    assert not r.low_confidence
    assert r.segment_id == 7 and r.n_sentences == 1
    sources, kw = tr.calls[0]
    assert sources[0][0] == "hye_Armn" and sources[0][-1] == "</s>"
    assert kw["target_prefix"] == [["eng_Latn"]]
    assert kw["max_decoding_length"] <= 256


def test_translate_batches_sentences():
    mt, tr = _engine({"Բարև։ Ինչպե՞ս ես։": "Hello. How are you?", "Ես լավ եմ։": "I'm fine."})
    r = mt.translate("Բարև։ Ինչպե՞ս ես։ Ես լավ եմ։")
    assert r.text == "Hello. How are you? I'm fine."
    assert r.n_sentences == 2
    assert len(tr.calls) == 1 and len(tr.calls[0][0]) == 2  # one batched call


def test_translate_empty_and_low_confidence():
    mt, _ = _engine({"Ինչ որ բան": "something"}, score=-2.5)
    assert not mt.translate("  ։ ").ok
    r = mt.translate("Ինչ որ բան")
    assert r.ok and r.low_confidence and "low confidence" in r.reason


def test_translate_stt_result():
    mt, _ = _engine({"Շնորհակալ եմ։": "Thank you."})
    stt = SimpleNamespace(text="Շնորհակալ եմ։", segment_id=3)
    r = mt.translate_stt(stt)
    assert r.text == "Thank you." and r.segment_id == 3
