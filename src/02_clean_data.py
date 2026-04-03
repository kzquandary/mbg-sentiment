import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common import SEED, ensure_dir, set_global_seed, write_json


VALID_LABELS = {"Positif", "Negatif", "Netral"}
LABEL_MAPPING = {
    "positif": "Positif",
    "negative": "Negatif",
    "negatif": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
    "netrall": "Netral",
}


def normalize_text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value: Any) -> str:
    raw = normalize_text_value(value)
    if not raw:
        return ""
    key = raw.lower()
    return LABEL_MAPPING.get(key, raw.title())


def backup_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_{timestamp}.bak{path.suffix}")
    path.rename(backup_path)
    return str(backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2 - Cleaning dataset")
    parser.add_argument("--input", type=str, default="data/dataset_relabel_mbg_improved_v2_boost.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--output-xlsx", type=str, default="data/dataset_clean.xlsx")
    parser.add_argument("--output-csv", type=str, default="data/dataset_clean.csv")
    parser.add_argument("--log-output", type=str, default="outputs/cleaning_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    output_xlsx = Path(args.output_xlsx)
    output_csv = Path(args.output_csv)
    log_output = Path(args.log_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ensure_dir(output_xlsx.parent)
    ensure_dir(output_csv.parent)
    ensure_dir(log_output.parent)

    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    n_before = int(len(df))
    raw_label_dist = {str(k): int(v) for k, v in df[args.label_col].astype(str).value_counts().to_dict().items()}

    df["_text_clean_strip"] = df[args.text_col].apply(normalize_text_value)
    empty_text_mask = df["_text_clean_strip"] == ""
    removed_empty_text = int(empty_text_mask.sum())
    df = df.loc[~empty_text_mask].copy()

    df["_label_standardized"] = df[args.label_col].apply(normalize_label)
    invalid_label_mask = ~df["_label_standardized"].isin(VALID_LABELS)
    removed_invalid_label = int(invalid_label_mask.sum())
    invalid_label_values = sorted(df.loc[invalid_label_mask, "_label_standardized"].dropna().unique().tolist())
    df = df.loc[~invalid_label_mask].copy()

    before_dedup = int(len(df))
    dedup_mask = df.duplicated(subset=["_text_clean_strip"], keep="first")
    removed_duplicate_text = int(dedup_mask.sum())
    df = df.loc[~dedup_mask].copy()

    df[args.text_col] = df["_text_clean_strip"]
    df[args.label_col] = df["_label_standardized"]
    df = df.drop(columns=["_text_clean_strip", "_label_standardized"])

    n_after = int(len(df))
    final_label_dist = {str(k): int(v) for k, v in df[args.label_col].value_counts().to_dict().items()}

    backup_info = {
        "output_xlsx_backup": backup_if_exists(output_xlsx),
        "output_csv_backup": backup_if_exists(output_csv),
        "log_output_backup": backup_if_exists(log_output),
    }

    df.to_excel(output_xlsx, index=False)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    cleaning_log = {
        "seed": SEED,
        "input_file": str(input_path),
        "text_column": args.text_col,
        "label_column": args.label_col,
        "raw_rows": n_before,
        "rows_after_remove_empty_text": n_before - removed_empty_text,
        "rows_before_dedup": before_dedup,
        "rows_after_cleaning": n_after,
        "removed_counts": {
            "empty_text": removed_empty_text,
            "invalid_label": removed_invalid_label,
            "duplicate_text_exact": removed_duplicate_text,
            "total_removed": n_before - n_after,
        },
        "label_distribution_before": raw_label_dist,
        "label_distribution_after": final_label_dist,
        "invalid_label_values_after_standardization": invalid_label_values,
        "label_mapping_rules": LABEL_MAPPING,
        "valid_labels": sorted(list(VALID_LABELS)),
        "backup_info": backup_info,
        "timestamp": datetime.now().isoformat(),
    }
    write_json(cleaning_log, log_output)

    print(f"[OK] Cleaned Excel: {output_xlsx}")
    print(f"[OK] Cleaned CSV: {output_csv}")
    print(f"[OK] Cleaning log: {log_output}")
    print(f"[INFO] Rows before: {n_before} | Rows after: {n_after}")


if __name__ == "__main__":
    main()
