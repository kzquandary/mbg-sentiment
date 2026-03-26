import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from utils.common import SEED, ensure_dir, set_global_seed, write_json


TEXT_CANDIDATES = ["text", "komentar", "comment", "content", "isi", "caption"]
LABEL_CANDIDATES = ["labeling_sentimen", "label", "sentimen", "sentiment", "kelas"]


def normalize_col_name(col: str) -> str:
    return (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(c): c for c in columns}
    for candidate in candidates:
        for norm, original in normalized_to_original.items():
            if candidate in norm:
                return original
    return None


def read_first_sheet(excel_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    active_sheet = sheet_names[0]
    df = pd.read_excel(excel_path, sheet_name=active_sheet)
    return df, {"sheet_names": sheet_names, "active_sheet": active_sheet}


def audit_dataset(df: pd.DataFrame, text_col: str | None, label_col: str | None) -> dict[str, Any]:
    missing_by_col = {k: int(v) for k, v in df.isna().sum().to_dict().items()}
    duplicate_rows = int(df.duplicated().sum())
    duplicate_text_rows = int(df.duplicated(subset=[text_col]).sum()) if text_col else None

    label_distribution_raw: dict[str, int] = {}
    label_inconsistency_candidates: dict[str, list[str]] = {}

    if label_col:
        label_series = df[label_col].astype("string")
        label_distribution_raw = {
            str(k): int(v)
            for k, v in label_series.fillna("<MISSING>").value_counts(dropna=False).to_dict().items()
        }

        grouped_variants: dict[str, set[str]] = {}
        for value in label_series.dropna().astype(str):
            stripped = value.strip()
            key = stripped.lower()
            grouped_variants.setdefault(key, set()).add(stripped)
        label_inconsistency_candidates = {
            key: sorted(list(values))
            for key, values in grouped_variants.items()
            if len(values) > 1
        }

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "text_col_detected": text_col,
        "label_col_detected": label_col,
        "missing_values_by_column": missing_by_col,
        "duplicate_full_rows": duplicate_rows,
        "duplicate_text_rows": duplicate_text_rows,
        "label_distribution_raw": label_distribution_raw,
        "label_inconsistency_candidates": label_inconsistency_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 - Audit dataset from .xlsx file.")
    parser.add_argument("--input", type=str, default="data/dataset.xlsx", help="Input xlsx path")
    parser.add_argument(
        "--summary-output", type=str, default="outputs/data_audit_summary.json", help="Audit summary JSON output"
    )
    parser.add_argument(
        "--preview-output", type=str, default="outputs/data_audit_preview.csv", help="Audit preview CSV output"
    )
    parser.add_argument("--preview-rows", type=int, default=50, help="Preview row count")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    summary_path = Path(args.summary_output)
    preview_path = Path(args.preview_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ensure_dir(summary_path.parent)
    ensure_dir(preview_path.parent)

    df, excel_meta = read_first_sheet(input_path)
    text_col = detect_column(list(df.columns), TEXT_CANDIDATES)
    label_col = detect_column(list(df.columns), LABEL_CANDIDATES)

    audit_result = audit_dataset(df, text_col=text_col, label_col=label_col)
    summary = {
        "seed": SEED,
        "input_file": str(input_path),
        "excel_info": excel_meta,
        "audit": audit_result,
    }
    write_json(summary, summary_path)

    df.head(args.preview_rows).to_csv(preview_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Summary saved: {summary_path}")
    print(f"[OK] Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
