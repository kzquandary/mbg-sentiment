import argparse
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate template review relabel dari kandidat audit.")
    parser.add_argument("--candidate-input", type=str, default="outputs/label_audit_candidates.csv")
    parser.add_argument("--output-xlsx", type=str, default="outputs/relabel_review_template.xlsx")
    parser.add_argument("--output-json", type=str, default="outputs/relabel_review_template_summary.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    candidate_input = Path(args.candidate_input)
    output_xlsx = Path(args.output_xlsx)
    output_json = Path(args.output_json)

    if not candidate_input.exists():
        raise FileNotFoundError(f"Candidate input not found: {candidate_input}")

    for p in [output_xlsx, output_json]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    cand = pd.read_csv(candidate_input)
    required_cols = {"text", "Labeling_Sentimen", "candidate_reason", "text_norm"}
    if not required_cols.issubset(cand.columns):
        raise KeyError(f"Candidate file missing required columns: {required_cols - set(cand.columns)}")

    review = cand.copy()
    review["row_id"] = range(1, len(review) + 1)
    review["current_label"] = review["Labeling_Sentimen"].astype(str)
    review["review_action"] = ""  # keep | relabel | drop
    review["reviewed_label"] = ""  # Positif | Negatif | Netral (if relabel)
    review["reviewer_note"] = ""

    # Urutan kolom agar nyaman direview manual
    ordered_cols = [
        "row_id",
        "text",
        "text_norm",
        "candidate_reason",
        "current_label",
        "review_action",
        "reviewed_label",
        "reviewer_note",
    ]
    review = review[ordered_cols]

    review.to_excel(output_xlsx, index=False)

    summary = {
        "seed": SEED,
        "candidate_input": str(candidate_input),
        "rows_template": int(len(review)),
        "candidate_reason_counts": {
            str(k): int(v) for k, v in review["candidate_reason"].value_counts().to_dict().items()
        },
        "allowed_review_action": ["keep", "relabel", "drop"],
        "allowed_reviewed_label": ["Positif", "Negatif", "Netral"],
        "output_xlsx": str(output_xlsx),
    }
    write_json(summary, output_json)

    print(f"[OK] Relabel template saved: {output_xlsx}")
    print(f"[OK] Template summary saved: {output_json}")
    print(f"[INFO] Rows to review: {len(review)}")


if __name__ == "__main__":
    main()
