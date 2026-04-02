import argparse
import re
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


POS_CUES = {
    "bagus",
    "baik",
    "mantap",
    "keren",
    "suka",
    "setuju",
    "membantu",
    "terbantu",
    "bermanfaat",
    "berhasil",
    "hebat",
    "top",
    "apresiasi",
    "luar biasa",
}

NEG_CUES = {
    "buruk",
    "jelek",
    "gagal",
    "bohong",
    "hoax",
    "ribet",
    "susah",
    "parah",
    "kecewa",
    "marah",
    "aneh",
    "ngawur",
    "korup",
    "tidak setuju",
    "ga setuju",
    "tidak membantu",
    "ga membantu",
}

NEGATORS = {"tidak", "bukan", "ga", "gak", "nggak", "tak", "enggak"}
VALID_LABELS = {"Positif", "Negatif", "Netral"}


def normalize_text(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cue_score(text_norm: str, cues: set[str]) -> int:
    score = 0
    for cue in cues:
        if " " in cue:
            if cue in text_norm:
                score += 2
        else:
            if re.search(rf"\b{re.escape(cue)}\b", text_norm):
                score += 1
    return score


def negation_flip_adjust(tokens: list[str], pos_score: int, neg_score: int) -> tuple[int, int]:
    # If a negator appears just before positive cue, shift weight toward negative and vice versa.
    for i in range(len(tokens) - 1):
        if tokens[i] in NEGATORS:
            nxt = tokens[i + 1]
            if nxt in POS_CUES:
                pos_score = max(0, pos_score - 1)
                neg_score += 1
            if nxt in NEG_CUES:
                neg_score = max(0, neg_score - 1)
                pos_score += 1
    return pos_score, neg_score


def decide_label(pos_score: int, neg_score: int, margin_threshold: int) -> tuple[str, int]:
    margin = pos_score - neg_score
    if margin >= margin_threshold:
        return "Positif", margin
    if margin <= -margin_threshold:
        return "Negatif", margin
    return "Netral", margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused auto-relabel on high-confidence subset.")
    parser.add_argument("--input", type=str, default="data/dataset_relabel_mbg_improved.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--margin-threshold", type=int, default=2)
    parser.add_argument("--max-changes", type=int, default=420)
    parser.add_argument("--max-netral-to-negatif", type=int, default=200)
    parser.add_argument("--max-netral-to-positif", type=int, default=160)
    parser.add_argument("--max-negatif-to-positif", type=int, default=30)
    parser.add_argument("--max-positif-to-negatif", type=int, default=30)
    parser.add_argument("--max-negatif-to-netral", type=int, default=120)
    parser.add_argument("--max-positif-to-netral", type=int, default=120)
    parser.add_argument("--output", type=str, default="data/dataset_relabel_mbg_improved_v2.csv")
    parser.add_argument("--candidate-output", type=str, default="outputs/relabel_focus_candidates.csv")
    parser.add_argument("--log-output", type=str, default="outputs/relabel_focus_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    input_path = Path(args.input)
    out_path = Path(args.output)
    cand_path = Path(args.candidate_output)
    log_path = Path(args.log_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    for p in [out_path, cand_path, log_path]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise KeyError("Text/label column not found.")

    work = df.copy()
    work[args.text_col] = work[args.text_col].fillna("").astype(str)
    work[args.label_col] = work[args.label_col].fillna("").astype(str).str.strip()
    work = work[work[args.label_col].isin(VALID_LABELS)].copy()

    work["text_norm"] = work[args.text_col].map(normalize_text)
    work["pos_score"] = work["text_norm"].map(lambda x: cue_score(x, POS_CUES))
    work["neg_score"] = work["text_norm"].map(lambda x: cue_score(x, NEG_CUES))

    adjusted = []
    for _, r in work.iterrows():
        tokens = str(r["text_norm"]).split()
        p, n = negation_flip_adjust(tokens, int(r["pos_score"]), int(r["neg_score"]))
        adjusted.append((p, n))
    work["pos_score_adj"] = [p for p, _ in adjusted]
    work["neg_score_adj"] = [n for _, n in adjusted]

    decisions = work.apply(
        lambda r: decide_label(int(r["pos_score_adj"]), int(r["neg_score_adj"]), args.margin_threshold), axis=1
    )
    work["auto_label"] = [x[0] for x in decisions]
    work["margin"] = [int(x[1]) for x in decisions]
    work["confidence"] = (work["pos_score_adj"] - work["neg_score_adj"]).abs()

    candidates = work[work["auto_label"] != work[args.label_col]].copy()
    candidates = candidates.sort_values(
        by=["confidence", "pos_score_adj", "neg_score_adj"], ascending=[False, False, False]
    ).reset_index()
    candidates["from_to"] = candidates[args.label_col] + "->" + candidates["auto_label"]

    quotas = {
        "Netral->Negatif": int(args.max_netral_to_negatif),
        "Netral->Positif": int(args.max_netral_to_positif),
        "Negatif->Positif": int(args.max_negatif_to_positif),
        "Positif->Negatif": int(args.max_positif_to_negatif),
        "Negatif->Netral": int(args.max_negatif_to_netral),
        "Positif->Netral": int(args.max_positif_to_netral),
    }
    used = {k: 0 for k in quotas}
    selected_idx = []

    for _, row in candidates.iterrows():
        if len(selected_idx) >= int(args.max_changes):
            break
        ft = str(row["from_to"])
        if ft in quotas:
            if used[ft] >= quotas[ft]:
                continue
            used[ft] += 1
            selected_idx.append(int(row["index"]))

    before_counts = work[args.label_col].value_counts().to_dict()
    out = work.copy()
    out.loc[selected_idx, args.label_col] = out.loc[selected_idx, "auto_label"]
    after_counts = out[args.label_col].value_counts().to_dict()

    # Persist minimal columns from original dataset while preserving row count/order for valid labels only.
    merged = df.copy()
    merged_idx = out.index
    merged.loc[merged_idx, args.label_col] = out[args.label_col].values
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    cand_cols = [
        "index",
        args.text_col,
        args.label_col,
        "auto_label",
        "from_to",
        "confidence",
        "pos_score_adj",
        "neg_score_adj",
        "margin",
    ]
    out_candidates = candidates[cand_cols].copy()
    out_candidates["selected_for_apply"] = out_candidates["index"].isin(selected_idx)
    out_candidates.to_csv(cand_path, index=False, encoding="utf-8-sig")

    payload = {
        "seed": SEED,
        "input_file": str(input_path),
        "output_file": str(out_path),
        "rows_total": int(len(df)),
        "rows_valid_labels": int(len(work)),
        "margin_threshold": int(args.margin_threshold),
        "max_changes": int(args.max_changes),
        "candidates_found": int(len(candidates)),
        "changes_applied": int(len(selected_idx)),
        "quota_requested": quotas,
        "quota_used": used,
        "label_distribution_before": {str(k): int(v) for k, v in before_counts.items()},
        "label_distribution_after": {str(k): int(v) for k, v in after_counts.items()},
    }
    write_json(payload, log_path)

    print(f"[OK] Relabel v2 saved: {out_path}")
    print(f"[OK] Candidate table saved: {cand_path}")
    print(f"[OK] Relabel log saved: {log_path}")
    print(f"[INFO] Candidates: {len(candidates)} | Applied: {len(selected_idx)}")
    print(f"[INFO] Label dist before: {before_counts}")
    print(f"[INFO] Label dist after: {after_counts}")


if __name__ == "__main__":
    main()
