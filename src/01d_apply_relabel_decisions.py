import argparse
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


VALID_ACTION = {"keep", "relabel", "drop", ""}
VALID_LABELS = {"Positif", "Negatif", "Netral"}


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply keputusan relabel dari file review.")
    parser.add_argument("--input-dataset", type=str, default="data/cleaned_dataset.csv")
    parser.add_argument("--review-file", type=str, default="outputs/relabel_review_template.xlsx")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--output-csv", type=str, default="data/cleaned_dataset_relabelled.csv")
    parser.add_argument("--output-xlsx", type=str, default="data/cleaned_dataset_relabelled.xlsx")
    parser.add_argument("--log-output", type=str, default="outputs/relabel_apply_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_dataset = Path(args.input_dataset)
    review_file = Path(args.review_file)
    output_csv = Path(args.output_csv)
    output_xlsx = Path(args.output_xlsx)
    log_output = Path(args.log_output)

    if not input_dataset.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dataset}")
    if not review_file.exists():
        raise FileNotFoundError(f"Review file not found: {review_file}")

    for p in [output_csv, output_xlsx, log_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_dataset)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise KeyError("Text/label column not found in dataset.")

    rev = pd.read_excel(review_file)
    required_cols = {"text_norm", "review_action", "reviewed_label", "current_label"}
    if not required_cols.issubset(rev.columns):
        raise KeyError(f"Review file missing required columns: {required_cols - set(rev.columns)}")

    rev = rev.copy()
    rev["review_action"] = rev["review_action"].fillna("").astype(str).str.strip().str.lower()
    rev["reviewed_label"] = rev["reviewed_label"].fillna("").astype(str).str.strip()
    rev["text_norm"] = rev["text_norm"].astype(str).map(normalize_text)

    invalid_actions = sorted(set(rev["review_action"].unique()) - VALID_ACTION)
    if invalid_actions:
        raise ValueError(f"Invalid review_action values: {invalid_actions}")

    invalid_relabels = rev[
        (rev["review_action"] == "relabel") & (~rev["reviewed_label"].isin(VALID_LABELS))
    ]
    if len(invalid_relabels) > 0:
        raise ValueError("Ada baris relabel tanpa reviewed_label valid (Positif/Negatif/Netral).")

    # Ambil keputusan terakhir per text_norm (kalau ada duplikasi di file review)
    rev_last = rev.groupby("text_norm", as_index=False).tail(1)
    action_map = {
        row["text_norm"]: (row["review_action"], row["reviewed_label"])
        for _, row in rev_last.iterrows()
        if row["review_action"] in {"relabel", "drop"}
    }

    df = df.copy()
    df["text_norm"] = df[args.text_col].fillna("").astype(str).map(normalize_text)

    relabel_count = 0
    drop_count = 0
    touched_rows = 0

    keep_mask = [True] * len(df)
    new_labels = df[args.label_col].astype(str).tolist()

    for i, row in df.reset_index(drop=True).iterrows():
        key = row["text_norm"]
        if key not in action_map:
            continue
        action, relabel = action_map[key]
        touched_rows += 1
        if action == "drop":
            keep_mask[i] = False
            drop_count += 1
        elif action == "relabel":
            if new_labels[i] != relabel:
                new_labels[i] = relabel
                relabel_count += 1

    df[args.label_col] = new_labels
    out_df = df.loc[keep_mask].copy().drop(columns=["text_norm"])

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    out_df.to_excel(output_xlsx, index=False)

    log_payload = {
        "seed": SEED,
        "input_dataset": str(input_dataset),
        "review_file": str(review_file),
        "rows_before": int(len(df)),
        "rows_after": int(len(out_df)),
        "rows_touched_by_review": int(touched_rows),
        "rows_relabelled": int(relabel_count),
        "rows_dropped": int(drop_count),
        "final_label_distribution": {
            str(k): int(v) for k, v in out_df[args.label_col].value_counts().to_dict().items()
        },
        "output_csv": str(output_csv),
        "output_xlsx": str(output_xlsx),
    }
    write_json(log_payload, log_output)

    print(f"[OK] Relabelled dataset saved: {output_csv}")
    print(f"[OK] Relabelled dataset saved: {output_xlsx}")
    print(f"[OK] Relabel apply log saved: {log_output}")
    print(f"[INFO] Relabelled rows: {relabel_count} | Dropped rows: {drop_count} | Final rows: {len(out_df)}")


if __name__ == "__main__":
    main()
