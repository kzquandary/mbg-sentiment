import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


POSITIVE_WORDS = {
    "bagus",
    "baik",
    "setuju",
    "dukung",
    "mendukung",
    "mantap",
    "hebat",
    "keren",
    "sukses",
    "berhasil",
    "bener",
    "benar",
    "oke",
    "ok",
    "top",
    "sepakat",
    "tepat",
}

NEGATIVE_WORDS = {
    "buruk",
    "gagal",
    "tolak",
    "stop",
    "hentikan",
    "racun",
    "keracunan",
    "basi",
    "korup",
    "jelek",
    "salah",
    "bohong",
    "parah",
    "mending",
    "alih",
    "alihkan",
    "sabotase",
    "makar",
}


def normalize_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def weak_label_from_text(text_norm: str) -> tuple[str | None, int, str]:
    tokens = text_norm.split()
    pos_score = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg_score = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    rule = "none"

    if (
        "jangan salahkan presiden" in text_norm
        or "program pak prabowo baik" in text_norm
        or ("niat" in text_norm and "baik" in text_norm)
    ):
        pos_score += 2
        rule = "phrase_positive_boost"

    if "lebih baik" in text_norm and "sekolah" in text_norm and "gratis" in text_norm:
        neg_score += 2
        rule = "phrase_negative_boost"

    if "stop mbg" in text_norm:
        neg_score += 2
        rule = "stop_mbg_negative_boost"

    if pos_score >= neg_score + 1 and pos_score >= 1:
        return "Positif", int(pos_score - neg_score), "lexicon_positive" if rule == "none" else rule

    if neg_score >= pos_score + 1 and neg_score >= 1:
        return "Negatif", int(neg_score - pos_score), "lexicon_negative" if rule == "none" else rule

    if pos_score == 0 and neg_score == 0 and len(tokens) <= 5:
        return "Netral", 1, "short_no_polar_words"

    return None, 0, "ambiguous_or_low_signal"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare demo dataset with transparent weak-supervision relabeling."
    )
    parser.add_argument("--input", type=str, default="data/dataset_clean.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--keep-unassigned", action="store_true")
    parser.add_argument("--output-csv", type=str, default="data/dataset_demo_adjusted.csv")
    parser.add_argument("--output-xlsx", type=str, default="data/dataset_demo_adjusted.xlsx")
    parser.add_argument("--log-output", type=str, default="outputs/demo_adjustment_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_xlsx = Path(args.output_xlsx)
    log_output = Path(args.log_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for p in [output_csv, output_xlsx, log_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    out = df.copy()
    out["text_norm_demo"] = out[args.text_col].fillna("").astype(str).apply(normalize_text)

    assigned_labels = []
    confidence_scores = []
    weak_rules = []
    for text_norm in out["text_norm_demo"].tolist():
        lbl, conf, rule = weak_label_from_text(text_norm)
        assigned_labels.append(lbl)
        confidence_scores.append(conf)
        weak_rules.append(rule)

    out["label_original"] = out[args.label_col].astype(str)
    out["label_weak"] = assigned_labels
    out["weak_confidence"] = confidence_scores
    out["weak_rule"] = weak_rules

    if args.keep_unassigned:
        out["label_demo_final"] = out["label_weak"].fillna(out["label_original"])
        kept = out.copy()
        dropped_rows = 0
    else:
        keep_mask = out["label_weak"].notna() & (out["weak_confidence"] >= int(args.min_confidence))
        kept = out.loc[keep_mask].copy()
        kept["label_demo_final"] = kept["label_weak"]
        dropped_rows = int((~keep_mask).sum())

    kept[args.label_col] = kept["label_demo_final"]
    kept = kept.drop(columns=["label_demo_final"])

    kept.to_csv(output_csv, index=False, encoding="utf-8-sig")
    kept.to_excel(output_xlsx, index=False)

    log_payload = {
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "output_csv": str(output_csv),
        "output_xlsx": str(output_xlsx),
        "text_column": args.text_col,
        "label_column": args.label_col,
        "weak_labeling_mode": {
            "min_confidence": int(args.min_confidence),
            "keep_unassigned": bool(args.keep_unassigned),
        },
        "rows_before": int(len(df)),
        "rows_after": int(len(kept)),
        "rows_dropped": dropped_rows,
        "label_distribution_original": {
            str(k): int(v) for k, v in out["label_original"].value_counts().to_dict().items()
        },
        "label_distribution_demo_final": {
            str(k): int(v) for k, v in kept[args.label_col].value_counts().to_dict().items()
        },
        "weak_rule_counts_all_rows": {
            str(k): int(v) for k, v in out["weak_rule"].value_counts().to_dict().items()
        },
        "weak_assigned_counts": {
            str(k): int(v)
            for k, v in out["label_weak"].fillna("<UNASSIGNED>").value_counts(dropna=False).to_dict().items()
        },
        "note": (
            "Dataset demo disusun dengan weak-supervision rules yang eksplisit untuk pembelajaran."
            " Bukan evaluasi ilmiah final."
        ),
    }
    write_json(log_payload, log_output)

    print(f"[OK] Demo adjusted CSV saved: {output_csv}")
    print(f"[OK] Demo adjusted XLSX saved: {output_xlsx}")
    print(f"[OK] Demo adjustment log saved: {log_output}")
    print(f"[INFO] Rows before: {len(df)} | Rows after: {len(kept)} | Rows dropped: {dropped_rows}")


if __name__ == "__main__":
    main()
