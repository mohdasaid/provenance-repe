"""Tokenizer fertility: subword tokens per whitespace word.

Two lines of code, and it contextualises every other number in the paper.
Also the bridge to the follow-up work on extraction-position validity: if a
Hausa word costs 4 tokens and an English one costs 1.3, "the last token"
means something different in each language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fertility(texts: list[str], tok) -> dict:
    words, toks = 0, 0
    per_text = []
    for t in texts:
        w = len(t.split())
        n = len(tok.encode(t, add_special_tokens=False))
        words += w
        toks += n
        per_text.append(n / max(w, 1))
    return {
        "tokens_per_word": toks / max(words, 1),
        "median": float(np.median(per_text)),
        "n_texts": len(texts),
    }


def table(df: pd.DataFrame, tok) -> pd.DataFrame:
    """Fertility by language and arm."""
    rows = []
    for (lang, arm), g in df.groupby(["lang", "arm"]):
        texts = g["positive"].tolist() + g["negative"].tolist()
        f = fertility(texts, tok)
        rows.append({"lang": lang, "arm": arm, **f})
    return pd.DataFrame(rows)
