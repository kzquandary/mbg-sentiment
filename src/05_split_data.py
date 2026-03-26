import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


def label_distribution(df: pd.DataFrame, label_col: str) -> dict[str, dict[str, float]]:
    counts = df[label_col].value_counts().sort_index()
    total = int(len(df))
    result: dict[str, dict[str, float]] = {}
    for label, count in counts.items():
        result[str(label)] = {
            "count": int(count),
            "proportion": round(float(count) / float(total), 6) if total > 0 else 0.0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 - Stratified train/val/test split")
    parser.add_argument("--input", type=str, default="data/preprocessed_dataset.csv")
    parser.add_argument("--text-col", type=str, default="text_model_input")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-output", type=str, default="data/train.csv")
    parser.add_argument("--val-output", type=str, default="data/val.csv")
    parser.add_argument("--test-output", type=str, default="data/test.csv")
    parser.add_argument("--summary-output", type=str, default="outputs/split_summary.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1.0:
        raise ValueError("train-ratio + val-ratio + test-ratio harus = 1.0")

    input_path = Path(args.input)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    test_output = Path(args.test_output)
    summary_output = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ensure_dir(train_output.parent)
    ensure_dir(val_output.parent)
    ensure_dir(test_output.parent)
    ensure_dir(summary_output.parent)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    before_rows = int(len(df))
    text_clean = df[args.text_col].fillna("").astype(str).str.strip()
    empty_text_mask = text_clean == ""
    removed_empty_text = int(empty_text_mask.sum())
    df = df.loc[~empty_text_mask].copy()
    after_rows = int(len(df))

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - args.train_ratio),
        random_state=SEED,
        stratify=df[args.label_col],
    )

    relative_test_size = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=temp_df[args.label_col],
    )

    for p in [train_output, val_output, test_output, summary_output]:
        backup_if_exists(p)

    train_df.to_csv(train_output, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_output, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_output, index=False, encoding="utf-8-sig")

    summary = {
        "seed": SEED,
        "input_file": str(input_path),
        "text_column": args.text_col,
        "label_column": args.label_col,
        "split_ratio": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "cleaning_before_split": {
            "rows_before": before_rows,
            "removed_empty_text_model_input": removed_empty_text,
            "rows_after": after_rows,
        },
        "size": {
            "full_after_filter": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "distribution": {
            "full_after_filter": label_distribution(df, args.label_col),
            "train": label_distribution(train_df, args.label_col),
            "val": label_distribution(val_df, args.label_col),
            "test": label_distribution(test_df, args.label_col),
        },
    }
    write_json(summary, summary_output)

    print(f"[OK] Train saved: {train_output}")
    print(f"[OK] Val saved: {val_output}")
    print(f"[OK] Test saved: {test_output}")
    print(f"[OK] Split summary saved: {summary_output}")
    print(f"[INFO] Sizes -> train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")


if __name__ == "__main__":
    main()
