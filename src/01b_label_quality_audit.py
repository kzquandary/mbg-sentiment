import argparse
import re
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


def normalize_text(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit kualitas label (kandidat salah label).")
    parser.add_argument("--input", type=str, default="data/cleaned_dataset.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--out-csv", type=str, default="outputs/label_audit_candidates.csv")
    parser.add_argument("--out-json", type=str, default="outputs/label_audit_summary.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for p in [out_csv, out_json]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise KeyError("Kolom teks/label tidak ditemukan.")

    df = df.copy()
    df["text_norm"] = df[args.text_col].fillna("").astype(str).apply(normalize_text)
    df = df[df["text_norm"] != ""].copy()

    # Kandidat 1: teks sama persis tapi punya label berbeda
    grp = df.groupby("text_norm")[args.label_col].nunique().reset_index(name="n_unique_labels")
    conflict_texts = grp[grp["n_unique_labels"] > 1]["text_norm"]
    conflict_df = df[df["text_norm"].isin(conflict_texts)].copy()
    conflict_df["candidate_reason"] = "same_text_multiple_labels"

    # Kandidat 2: teks sangat pendek tapi berlabel ekstrem (risk ambiguous)
    df["n_tokens"] = df["text_norm"].str.split().str.len()
    short_df = df[(df["n_tokens"] <= 2) & (df[args.label_col].isin(["Positif", "Negatif"]))].copy()
    short_df["candidate_reason"] = "very_short_extreme_label"

    candidates = pd.concat([conflict_df, short_df], axis=0, ignore_index=True).drop_duplicates(
        subset=["text_norm", args.label_col, "candidate_reason"]
    )

    # Prioritasi: konflik label dulu
    reason_order = {"same_text_multiple_labels": 0, "very_short_extreme_label": 1}
    candidates["reason_rank"] = candidates["candidate_reason"].map(reason_order).fillna(9)
    candidates = candidates.sort_values(["reason_rank", "text_norm"]).drop(columns=["reason_rank"])

    cols = [c for c in [args.text_col, args.label_col, "candidate_reason", "text_norm", "n_tokens"] if c in candidates.columns]
    candidates[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED,
        "input_file": str(input_path),
        "rows_input": int(len(df)),
        "candidate_rows_total": int(len(candidates)),
        "candidate_by_reason": {
            k: int(v) for k, v in candidates["candidate_reason"].value_counts().to_dict().items()
        },
        "unique_conflict_texts": int(conflict_texts.nunique()),
        "output_csv": str(out_csv),
    }
    write_json(summary, out_json)

    print(f"[OK] Label audit candidates saved: {out_csv}")
    print(f"[OK] Label audit summary saved: {out_json}")
    print(f"[INFO] Total candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
