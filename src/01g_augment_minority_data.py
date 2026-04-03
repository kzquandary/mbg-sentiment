import argparse
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed, write_json


SYNONYM_MAP = {
    "bagus": ["baik", "mantap", "keren"],
    "baik": ["bagus", "mantap"],
    "mantap": ["bagus", "keren"],
    "keren": ["bagus", "mantap"],
    "suka": ["senang", "setuju"],
    "setuju": ["sepakat", "dukung"],
    "dukung": ["support", "setuju"],
    "membantu": ["bermanfaat", "menolong"],
    "bermanfaat": ["membantu", "berguna"],
    "buruk": ["jelek", "parah"],
    "jelek": ["buruk", "kurang bagus"],
    "gagal": ["tidak berhasil", "zonk"],
    "kecewa": ["sedih", "kurang puas"],
    "ribet": ["rumit", "repot"],
    "susah": ["sulit", "berat"],
    "parah": ["buruk", "fatal"],
    "bohong": ["tipu", "ngibul"],
    "hoax": ["bohong", "tipuan"],
}

PREFIX_POS = ["menurut saya", "jujur", "sejauh ini", "buat saya"]
PREFIX_NEG = ["jujur aja", "menurut saya", "buat saya", "kalau lihat ini"]
SUFFIX_POS = ["cukup memuaskan", "semoga konsisten", "lanjutkan"]
SUFFIX_NEG = ["tolong diperbaiki", "harap dievaluasi", "masih banyak PR"]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def synonym_replace(text: str, rng: random.Random) -> str:
    tokens = text.split()
    idxs = [i for i, tok in enumerate(tokens) if tok.lower() in SYNONYM_MAP]
    if not idxs:
        return text
    i = rng.choice(idxs)
    tok = tokens[i].lower()
    repl = rng.choice(SYNONYM_MAP[tok])
    if tokens[i][:1].isupper():
        repl = repl[:1].upper() + repl[1:]
    tokens[i] = repl
    return normalize_space(" ".join(tokens))


def mild_paraphrase(text: str, label: str, rng: random.Random) -> str:
    base = normalize_space(text)
    if not base:
        return base

    out = synonym_replace(base, rng)

    # Add lightweight discourse markers to create natural variants.
    if label == "Positif":
        if rng.random() < 0.35:
            out = f"{rng.choice(PREFIX_POS)}, {out}"
        if rng.random() < 0.25:
            out = f"{out}, {rng.choice(SUFFIX_POS)}"
    elif label == "Negatif":
        if rng.random() < 0.35:
            out = f"{rng.choice(PREFIX_NEG)}, {out}"
        if rng.random() < 0.25:
            out = f"{out}, {rng.choice(SUFFIX_NEG)}"

    if rng.random() < 0.2 and not out.endswith(("!", ".", "?")):
        out = out + "."
    return normalize_space(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment minority sentiment classes with lightweight paraphrase.")
    parser.add_argument("--input", type=str, default="data/dataset_relabel_mbg_improved_v2_boost.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--target-negatif", type=int, default=1100)
    parser.add_argument("--target-positif", type=int, default=900)
    parser.add_argument("--output", type=str, default="data/dataset_relabel_mbg_improved_v3_aug.csv")
    parser.add_argument("--log-output", type=str, default="outputs/augment_v3_log.json")
    args = parser.parse_args()

    set_global_seed(SEED)
    rng = random.Random(SEED)

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_path = Path(args.log_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    for p in [output_path, log_path]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(input_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise KeyError(f"Required columns not found: {args.text_col}, {args.label_col}")

    base = df.copy()
    base[args.text_col] = base[args.text_col].fillna("").astype(str).map(normalize_space)
    base = base[base[args.text_col] != ""].copy()

    counts_before = base[args.label_col].value_counts().to_dict()
    cur_neg = int(counts_before.get("Negatif", 0))
    cur_pos = int(counts_before.get("Positif", 0))

    need_neg = max(0, int(args.target_negatif) - cur_neg)
    need_pos = max(0, int(args.target_positif) - cur_pos)

    aug_rows = []

    def build_aug(label: str, n_need: int) -> None:
        if n_need <= 0:
            return
        pool = base[base[args.label_col] == label]
        if pool.empty:
            return
        pool_idx = pool.index.tolist()
        for i in range(n_need):
            src_idx = pool_idx[i % len(pool_idx)]
            src = base.loc[src_idx].copy()
            original_text = str(src[args.text_col])
            new_text = mild_paraphrase(original_text, label, rng)
            if new_text == original_text:
                new_text = f"{original_text}."
            src[args.text_col] = new_text
            src["source_folder"] = str(src.get("source_folder", "original")) + "|aug_v3"
            src["cid"] = f"aug_v3_{label.lower()}_{i+1:05d}"
            src["augmentation_tag"] = "minority_paraphrase_v3"
            aug_rows.append(src)

    build_aug("Negatif", need_neg)
    build_aug("Positif", need_pos)

    aug_df = pd.DataFrame(aug_rows)
    if not aug_df.empty:
        # Align columns with base and keep augmentation tag.
        for col in base.columns:
            if col not in aug_df.columns:
                aug_df[col] = None
        out = pd.concat([base, aug_df[base.columns.tolist() + ["augmentation_tag"]]], ignore_index=True)
    else:
        out = base.copy()
        if "augmentation_tag" not in out.columns:
            out["augmentation_tag"] = None

    counts_after = out[args.label_col].value_counts().to_dict()
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    write_json(
        {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "target_negatif": int(args.target_negatif),
            "target_positif": int(args.target_positif),
            "before_distribution": {str(k): int(v) for k, v in counts_before.items()},
            "after_distribution": {str(k): int(v) for k, v in counts_after.items()},
            "added_rows": int(len(aug_df)),
            "added_negatif": int(need_neg),
            "added_positif": int(need_pos),
            "notes": "Augmentation uses lightweight paraphrase with synonym replacement and discourse markers.",
        },
        log_path,
    )

    print(f"[OK] Augmented dataset saved: {output_path}")
    print(f"[OK] Augmentation log saved: {log_path}")
    print(f"[INFO] Added rows: {len(aug_df)}")
    print(f"[INFO] Before: {counts_before}")
    print(f"[INFO] After : {counts_after}")


if __name__ == "__main__":
    main()
