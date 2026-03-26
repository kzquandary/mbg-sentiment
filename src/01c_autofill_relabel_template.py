import argparse
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


POSITIVE_HINTS = {
    "bagus",
    "mantap",
    "setuju",
    "benar",
    "betul",
    "oke",
    "baik",
    "dukung",
    "hebat",
    "keren",
    "top",
    "sukses",
}

NEGATIVE_HINTS = {
    "buruk",
    "parah",
    "gagal",
    "bohong",
    "jelek",
    "tolak",
    "salah",
    "korup",
    "basi",
    "racun",
    "keracunan",
}

NON_INFORMATIVE_SHORT = {
    "iya",
    "ya",
    "ok",
    "oke",
    "sip",
    "hmm",
    "hehe",
    "haha",
    "wkwk",
    "wkwkwk",
    "mantap",
    "betul",
    "benar",
    "sama",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Autofill relabel template with explicit rules.")
    parser.add_argument("--template", type=str, default="outputs/relabel_review_template.xlsx")
    parser.add_argument("--dataset", type=str, default="data/cleaned_dataset.csv")
    parser.add_argument("--output", type=str, default="outputs/relabel_review_autofilled.xlsx")
    parser.add_argument("--summary", type=str, default="outputs/relabel_autofill_summary.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    template_path = Path(args.template)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    for p in [output_path, summary_path]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    review = pd.read_excel(template_path)
    df = pd.read_csv(dataset_path)
    df["text_norm"] = df["text"].fillna("").astype(str).str.lower().str.split().str.join(" ")

    # Majority label map by normalized text
    majority_map = {}
    for text_norm, grp in df.groupby("text_norm"):
        vc = grp["Labeling_Sentimen"].astype(str).value_counts()
        top = vc.index.tolist()
        if len(top) == 0:
            continue
        if len(top) > 1 and vc.iloc[0] == vc.iloc[1]:
            majority_map[text_norm] = "Netral"
        else:
            majority_map[text_norm] = top[0]

    review = review.copy()
    review["review_action"] = review["review_action"].fillna("").astype(str).str.strip().str.lower()
    review["reviewed_label"] = review["reviewed_label"].fillna("").astype(str).str.strip()
    review["reviewer_note"] = review["reviewer_note"].fillna("").astype(str)

    rule_counts = {
        "conflict_majority_relabel": 0,
        "short_noninformative_relabel_netral": 0,
        "short_hint_positive_relabel": 0,
        "short_hint_negative_relabel": 0,
        "short_default_netral": 0,
    }

    for i, row in review.iterrows():
        reason = str(row.get("candidate_reason", ""))
        text_norm = str(row.get("text_norm", "")).strip().lower()
        text = str(row.get("text", "")).strip().lower()
        tokens = text_norm.split()

        action = "keep"
        label = str(row.get("current_label", ""))
        note = ""

        if reason == "same_text_multiple_labels":
            maj = majority_map.get(text_norm, "Netral")
            action = "relabel"
            label = maj
            note = "auto: conflict text -> majority label (tie->Netral)"
            rule_counts["conflict_majority_relabel"] += 1
        elif reason == "very_short_extreme_label":
            if text_norm in NON_INFORMATIVE_SHORT:
                action = "relabel"
                label = "Netral"
                note = "auto: very short non-informative -> Netral"
                rule_counts["short_noninformative_relabel_netral"] += 1
            elif any(tok in POSITIVE_HINTS for tok in tokens):
                action = "relabel"
                label = "Positif"
                note = "auto: short with positive hint"
                rule_counts["short_hint_positive_relabel"] += 1
            elif any(tok in NEGATIVE_HINTS for tok in tokens):
                action = "relabel"
                label = "Negatif"
                note = "auto: short with negative hint"
                rule_counts["short_hint_negative_relabel"] += 1
            else:
                action = "relabel"
                label = "Netral"
                note = "auto: short ambiguous default -> Netral"
                rule_counts["short_default_netral"] += 1

        review.at[i, "review_action"] = action
        review.at[i, "reviewed_label"] = label if action == "relabel" else ""
        review.at[i, "reviewer_note"] = note

    review.to_excel(output_path, index=False)

    action_counts = review["review_action"].value_counts().to_dict()
    relabel_counts = review.loc[review["review_action"] == "relabel", "reviewed_label"].value_counts().to_dict()
    payload = {
        "seed": SEED,
        "template_input": str(template_path),
        "dataset_input": str(dataset_path),
        "rows": int(len(review)),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "relabel_target_counts": {str(k): int(v) for k, v in relabel_counts.items()},
        "rule_counts": {str(k): int(v) for k, v in rule_counts.items()},
        "output_template": str(output_path),
    }
    write_json(payload, summary_path)

    print(f"[OK] Autofilled template saved: {output_path}")
    print(f"[OK] Autofill summary saved: {summary_path}")
    print(f"[INFO] Actions: {payload['action_counts']}")


if __name__ == "__main__":
    main()
