"""Load contrast pairs from the collection spreadsheets and validate them.

One tidy DataFrame is the contract for everything downstream:
    pair_id | lang | arm | subject | writer_id | positive | negative
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED = ["pair_id", "subject", "positive", "negative"]

# concept -> (sheet name, positive column header, negative column header)
CONCEPT_SHEETS = {
    "sentiment":  ("Pairs - Sentiment",  "pleased message",    "annoyed message"),
    "politeness": ("Pairs - Politeness", "respectful message", "casual message"),
}


def nfc(s) -> str:
    """Canonical Unicode form. Yoruba tone marks and Hausa hooked letters have
    two byte-level representations that look identical on screen; a tokenizer
    treats them as different. Normalise once, on load, always."""
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFC", s).strip()


def load_sheet(path: Path | str, lang: str, arm: str,
               concept: str = "sentiment") -> pd.DataFrame:
    """Read one concept tab of a collection workbook into the tidy contract."""
    sheet, pos_col, neg_col = CONCEPT_SHEETS[concept]
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={pos_col: "positive", neg_col: "negative"})

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Got {list(df.columns)}")

    if "writer_id" not in df.columns:
        df["writer_id"] = "unknown"

    df = df[df["pair_id"].astype(str).str.upper() != "EXAMPLE"]
    for col in ("positive", "negative", "subject", "pair_id", "writer_id"):
        df[col] = df[col].map(nfc)

    df = df[(df["positive"] != "") & (df["negative"] != "")].copy()
    df["lang"] = lang
    df["arm"] = arm
    df["concept"] = concept
    return df[["pair_id", "lang", "arm", "concept", "subject", "writer_id",
               "positive", "negative"]].reset_index(drop=True)


def available_concepts(path: Path | str) -> list[str]:
    """Which concept tabs actually exist in this workbook."""
    import openpyxl
    names = set(openpyxl.load_workbook(path, read_only=True).sheetnames)
    return [c for c, (sheet, _, _) in CONCEPT_SHEETS.items() if sheet in names]


def validate(df: pd.DataFrame, max_ratio: float = 2.0) -> pd.DataFrame:
    """Flag pairs that break the matching rules. Returns a report, does not drop.

    Length imbalance is the one that silently poisons a diff-in-means vector:
    if positives are systematically longer than negatives, the vector encodes
    length, not the concept.
    """
    rep = df.copy()
    rep["n_pos"] = rep["positive"].str.split().str.len()
    rep["n_neg"] = rep["negative"].str.split().str.len()
    rep["ratio"] = rep[["n_pos", "n_neg"]].max(axis=1) / rep[["n_pos", "n_neg"]].min(axis=1).clip(lower=1)
    rep["flag_length"] = rep["ratio"] > max_ratio
    rep["flag_identical"] = rep["positive"] == rep["negative"]
    rep["flag_nfc"] = [
        p != unicodedata.normalize("NFC", p) for p in rep["positive"]
    ]
    return rep


def summarise(rep: pd.DataFrame) -> str:
    n = len(rep)
    mean_pos, mean_neg = rep["n_pos"].mean(), rep["n_neg"].mean()
    lines = [
        f"pairs: {n}",
        f"mean words  positive={mean_pos:.1f}  negative={mean_neg:.1f}  "
        f"(gap={abs(mean_pos - mean_neg):.1f})",
        f"length-flagged: {int(rep['flag_length'].sum())}",
        f"identical: {int(rep['flag_identical'].sum())}",
        f"writers: {rep['writer_id'].nunique()}   subjects: {rep['subject'].nunique()}",
    ]
    if abs(mean_pos - mean_neg) > 2:
        lines.append("WARNING: systematic length gap between sides — this will "
                     "leak into the vector. Rebalance before extracting.")
    return "\n".join(lines)
