import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


LABEL_MAP = {
    "positif": "Positif",
    "positive": "Positif",
    "negatif": "Negatif",
    "negative": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
}
VALID_LABELS = {"Positif", "Negatif", "Netral"}


def clean_label(raw: str) -> str:
    key = str(raw).strip().lower()
    return LABEL_MAP.get(key, str(raw).strip())


def normalize_text(raw: str) -> str:
    return " ".join(str(raw).strip().split())


def pick_label_with_tiebreak(group: pd.DataFrame, label_col: str) -> str:
    labels = group[label_col].astype(str).tolist()
    counts = Counter(labels)
    top_count = max(counts.values())
    top_labels = sorted([lbl for lbl, cnt in counts.items() if cnt == top_count])
    if len(top_labels) == 1:
        return top_labels[0]

    if "diggCount" in group.columns:
        grp = group.copy()
        grp["diggCount"] = pd.to_numeric(grp["diggCount"], errors="coerce").fillna(0)
        top = grp.sort_values("diggCount", ascending=False).iloc[0]
        return str(top[label_col])

    return top_labels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and improve relabel CSV quality.")
    parser.add_argument("--input", type=str, default="data/dataset_relabel_mbg.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--output", type=str, default="data/dataset_relabel_mbg_improved.csv")
    parser.add_argument("--audit-output", type=str, default="outputs/dataset_improvement_audit.json")
    parser.add_argument("--conflict-preview-output", type=str, default="outputs/dataset_label_conflict_preview.csv")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    preview_path = Path(args.conflict_preview_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for p in [output_path, audit_path, preview_path]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns:
        raise KeyError(f"Text column not found: {args.text_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    rows_before = int(len(df))
    label_dist_before = df[args.label_col].astype(str).str.strip().value_counts().to_dict()

    df[args.text_col] = df[args.text_col].fillna("").astype(str)
    df[args.label_col] = df[args.label_col].fillna("").astype(str)

    empty_text_mask = df[args.text_col].str.strip() == ""
    removed_empty_text = int(empty_text_mask.sum())
    df = df.loc[~empty_text_mask].copy()

    before_label_clean = df[args.label_col].copy()
    df[args.label_col] = df[args.label_col].map(clean_label)
    label_fixed_count = int((before_label_clean != df[args.label_col]).sum())

    invalid_label_mask = ~df[args.label_col].isin(VALID_LABELS)
    removed_invalid_labels = int(invalid_label_mask.sum())
    df = df.loc[~invalid_label_mask].copy()

    df["_text_norm"] = df[args.text_col].map(normalize_text)

    duplicate_groups = df.groupby("_text_norm", dropna=False)
    duplicate_text_count = int((duplicate_groups.size() > 1).sum())

    conflict_rows = []
    kept_rows = []
    conflict_group_count = 0
    for text_norm, grp in duplicate_groups:
        grp = grp.copy()
        unique_labels = sorted(grp[args.label_col].astype(str).unique().tolist())
        resolved_label = pick_label_with_tiebreak(grp, args.label_col)

        if len(unique_labels) > 1:
            conflict_group_count += 1
            conflict_rows.append(
                {
                    "text_normalized": text_norm,
                    "rows_in_group": int(len(grp)),
                    "labels_found": " | ".join(unique_labels),
                    "resolved_label": resolved_label,
                }
            )

        rep = grp.iloc[0].copy()
        rep[args.label_col] = resolved_label
        rep[args.text_col] = text_norm
        kept_rows.append(rep)

    out = pd.DataFrame(kept_rows).drop(columns=["_text_norm"], errors="ignore")
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    rows_after = int(len(out))
    dedup_removed = int(len(df) - len(out))
    label_dist_after = out[args.label_col].astype(str).value_counts().to_dict()

    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(conflict_rows).to_csv(preview_path, index=False, encoding="utf-8-sig")

    audit = {
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "text_column": args.text_col,
        "label_column": args.label_col,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "removed_empty_text": removed_empty_text,
        "label_fixed_count": label_fixed_count,
        "removed_invalid_labels": removed_invalid_labels,
        "dedup_removed": dedup_removed,
        "duplicate_text_group_count": duplicate_text_count,
        "label_conflict_group_count": conflict_group_count,
        "label_distribution_before": {str(k): int(v) for k, v in label_dist_before.items()},
        "label_distribution_after": {str(k): int(v) for k, v in label_dist_after.items()},
        "notes": [
            "Conflict label pada teks duplikat diselesaikan dengan voting mayoritas.",
            "Jika voting seri, dipilih baris dengan diggCount tertinggi (jika tersedia).",
            "Dataset ini lebih bersih untuk split & training. Test set tetap dibuat dari split baru, tidak di-balance.",
        ],
    }
    write_json(audit, audit_path)

    print(f"[OK] Improved dataset saved: {output_path}")
    print(f"[OK] Audit summary saved: {audit_path}")
    print(f"[OK] Conflict preview saved: {preview_path}")
    print(f"[INFO] Rows before: {rows_before} | Rows after: {rows_after}")
    print(f"[INFO] Label dist before: {label_dist_before}")
    print(f"[INFO] Label dist after: {label_dist_after}")


if __name__ == "__main__":
    main()
