"""Machine-translate the English seeds into Arm C, or round-trip Arm A into Arm D.

    # Arm C: English seeds -> Hausa
    python scripts/03_translate.py --lang hau --arm C

    # Arm D: round-trip native Hausa through English and back
    python scripts/03_translate.py --lang hau --arm D

    # bigger/better model (use on GCP, not the laptop)
    python scripts/03_translate.py --lang yor --arm C --model facebook/nllb-200-3.3B

Writes data/raw/{lang}_{arm}.xlsx with the SAME sheet names and column headers
as the Arm A collection workbook, so one loader handles every arm.

Progress is cached to a .jsonl after every batch. If the run dies, rerun the
same command and it resumes instead of re-translating.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from openpyxl import Workbook  # noqa: E402

from prov.config import Config  # noqa: E402
from prov.data import nfc  # noqa: E402

NLLB_CODE = {"hau": "hau_Latn", "yor": "yor_Latn", "eng": "eng_Latn"}

SHEETS = {
    "sentiment":  ("Pairs - Sentiment",  "PLEASED message",    "ANNOYED message"),
    "politeness": ("Pairs - Politeness", "RESPECTFUL message", "CASUAL message"),
}


def load_translator(model_id: str, device: str = "cpu"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return model, tok


def translate(texts, model, tok, src: str, tgt: str, batch_size: int = 8,
              max_new_tokens: int = 128) -> list[str]:
    """NLLB translation. src/tgt are NLLB codes like 'eng_Latn'."""
    import torch

    tok.src_lang = src
    # transformers >=4.51 dropped lang_code_to_id; convert_tokens_to_ids works
    # on both old and new versions.
    bos = tok.convert_tokens_to_ids(tgt)
    if bos is None or bos == tok.unk_token_id:
        raise ValueError(f"tokenizer does not know language code {tgt!r}")

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 max_new_tokens=max_new_tokens, num_beams=4,
                                 no_repeat_ngram_size=3,
                                 repetition_penalty=1.1)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return [nfc(t) for t in out]


def write_workbook(path: Path, frames: dict) -> None:
    """frames: {concept: DataFrame with pair_id, subject, positive, negative}"""
    wb = Workbook()
    wb.remove(wb.active)
    for concept, (sheet, pos_h, neg_h) in SHEETS.items():
        if concept not in frames:
            continue
        df = frames[concept]
        ws = wb.create_sheet(sheet)
        ws.append(["pair_id", "subject", pos_h, neg_h, "writer_id", "notes"])
        for _, r in df.iterrows():
            ws.append([r["pair_id"], r["subject"], r["positive"], r["negative"],
                       r["writer_id"], ""])
        for col, w in zip("ABCDEF", (12, 28, 50, 50, 14, 20)):
            ws.column_dimensions[col].width = w
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hau", "yor"])
    ap.add_argument("--arm", required=True, choices=["C", "D"])
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="translate only the first N pairs (for a smoke test)")
    args = ap.parse_args()

    cfg = Config.load()
    tgt = NLLB_CODE[args.lang]
    eng = NLLB_CODE["eng"]

    # ---- gather source rows ------------------------------------------------
    frames_src = {}
    if args.arm == "C":
        seeds = pd.read_csv(cfg.raw_dir / "english_seeds.csv")
        for concept, g in seeds.groupby("concept"):
            frames_src[concept] = g.rename(columns={
                "seed_id": "pair_id",
                "positive_en": "positive",
                "negative_en": "negative",
            })[["pair_id", "subject", "positive", "negative"]]
    else:  # arm D: round-trip the native arm A text
        from prov.data import load_sheet
        src_path = cfg.raw_dir / f"{args.lang}_A.xlsx"
        if not src_path.exists():
            sys.exit(f"arm D needs {src_path} (native data) — not found")
        for concept, (sheet, _, _) in SHEETS.items():
            try:
                frames_src[concept] = load_sheet(src_path, args.lang, "A",
                                                 sheet_name=sheet)
            except Exception as e:
                print(f"skip {concept}: {e}")

    if args.limit:
        frames_src = {k: v.head(args.limit) for k, v in frames_src.items()}

    model, tok = load_translator(args.model, args.device)
    model_tag = args.model.split("/")[-1]

    cache_path = cfg.raw_dir / f".cache_{args.lang}_{args.arm}_{model_tag}.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["key"]] = rec["text"]
        print(f"resuming: {len(done)} strings already translated")

    def run(texts, keys, src, tgt):
        todo = [(k, t) for k, t in zip(keys, texts) if k not in done]
        if todo:
            new = translate([t for _, t in todo], model, tok, src, tgt,
                            batch_size=args.batch_size)
            with cache_path.open("a", encoding="utf-8") as f:
                for (k, _), text in zip(todo, new):
                    done[k] = text
                    f.write(json.dumps({"key": k, "text": text},
                                       ensure_ascii=False) + "\n")
        return [done[k] for k in keys]

    frames_out = {}
    for concept, df in frames_src.items():
        print(f"\n{args.lang} {args.arm} {concept}: {len(df)} pairs")
        for side in ("positive", "negative"):
            keys = [f"{concept}:{pid}:{side}" for pid in df["pair_id"]]
            if args.arm == "C":
                df[side] = run(df[side].tolist(), keys, eng, tgt)
            else:
                mid = run(df[side].tolist(),
                          [k + ":mid" for k in keys], tgt, eng)
                df[side] = run(mid, keys, eng, tgt)
        df["writer_id"] = model_tag
        frames_out[concept] = df

    out = cfg.raw_dir / f"{args.lang}_{args.arm}.xlsx"
    write_workbook(out, frames_out)
    print(f"\nwrote {out}")
    print("spot-check a few rows with a native speaker before extracting.")


if __name__ == "__main__":
    main()
