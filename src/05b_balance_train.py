import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance train split by oversampling minority classes.")
    parser.add_argument("--input", type=str, default="data/train.csv")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--target-mode", type=str, default="max", choices=["max", "median", "fixed"])
    parser.add_argument("--target-count", type=int, default=0, help="Used only when target-mode=fixed")
    parser.add_argument("--output", type=str, default="data/train_balanced.csv")
    parser.add_argument("--log-output", type=str, default="outputs/train_balance_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for p in [output_path, log_path]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    label_counts = df[args.label_col].astype(str).value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError("Need at least 2 labels to balance.")

    counts_series = pd.Series(label_counts)
    if args.target_mode == "max":
        target_count = int(counts_series.max())
    elif args.target_mode == "median":
        target_count = int(counts_series.median())
    else:
        if args.target_count <= 0:
            raise ValueError("--target-count must be > 0 for target-mode=fixed")
        target_count = int(args.target_count)

    parts: list[pd.DataFrame] = []
    resample_plan: dict[str, dict[str, int]] = {}

    for label, grp in df.groupby(args.label_col):
        grp = grp.copy()
        n = int(len(grp))
        if n == target_count:
            sampled = grp
            action = "keep"
        elif n > target_count:
            sampled = grp.sample(n=target_count, random_state=SEED, replace=False)
            action = "downsample"
        else:
            extra = grp.sample(n=(target_count - n), random_state=SEED, replace=True)
            sampled = pd.concat([grp, extra], ignore_index=True)
            action = "oversample"
        parts.append(sampled)
        resample_plan[str(label)] = {"before": n, "after": int(len(sampled)), "action": action}

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    payload = {
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "label_column": args.label_col,
        "target_mode": args.target_mode,
        "target_count": target_count,
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
        "label_distribution_before": {str(k): int(v) for k, v in label_counts.items()},
        "label_distribution_after": {
            str(k): int(v) for k, v in out[args.label_col].astype(str).value_counts().to_dict().items()
        },
        "resample_plan": resample_plan,
        "note": "Balancing applied on train split only. Test split must remain untouched.",
    }
    write_json(payload, log_path)

    print(f"[OK] Balanced train saved: {output_path}")
    print(f"[OK] Balance log saved: {log_path}")
    print(f"[INFO] Rows before: {len(df)} | Rows after: {len(out)} | Target count per class: {target_count}")


if __name__ == "__main__":
    main()

