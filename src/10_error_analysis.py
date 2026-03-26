import argparse
import re
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


ENGLISH_HINT_WORDS = {
    "good",
    "bad",
    "nice",
    "school",
    "free",
    "program",
    "government",
    "food",
    "children",
    "healthy",
    "quality",
}


def detect_error_pattern(text: str) -> str:
    t = str(text).lower()
    tokens = t.split()
    if re.search(r"\b(wkwk|wk|haha|hehe|lol|sarkas|ironi)\b", t):
        return "indikasi_ironi_sarkasme"
    if any(tok in ENGLISH_HINT_WORDS for tok in tokens):
        return "bahasa_campuran"
    if re.search(r"(.)\1{2,}", t):
        return "typo_karakter_berulang"
    if len(tokens) <= 4 or "?" in t:
        return "konteks_ambigu_pendek"
    return "lainnya"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10 - Error analysis")
    parser.add_argument("--pred-input", type=str, default="outputs/test_predictions.csv")
    parser.add_argument("--text-col", type=str, default="text_original")
    parser.add_argument("--fallback-text-col", type=str, default="text_model_input")
    parser.add_argument("--sample-output", type=str, default="outputs/error_analysis_samples.xlsx")
    parser.add_argument("--summary-output", type=str, default="outputs/error_analysis_summary.md")
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    set_global_seed(SEED)

    pred_input = Path(args.pred_input)
    sample_output = Path(args.sample_output)
    summary_output = Path(args.summary_output)

    if not pred_input.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_input}")

    for p in [sample_output, summary_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    df = pd.read_csv(pred_input)
    required_cols = {"y_true", "y_pred", "is_correct"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"Kolom wajib tidak lengkap. Ditemukan: {df.columns.tolist()}")

    if args.text_col in df.columns:
        analysis_text_col = args.text_col
    elif args.fallback_text_col in df.columns:
        analysis_text_col = args.fallback_text_col
    else:
        raise KeyError("Kolom teks untuk error analysis tidak ditemukan.")

    err_df = df.loc[~df["is_correct"]].copy()
    if err_df.empty:
        err_df["error_pattern"] = []
    else:
        err_df["error_pattern"] = err_df[analysis_text_col].fillna("").astype(str).apply(detect_error_pattern)
    err_df["error_pair"] = err_df["y_true"].astype(str) + " -> " + err_df["y_pred"].astype(str)

    sample_df = err_df.head(args.max_samples).copy()
    sample_df.to_excel(sample_output, index=False)

    total = int(len(df))
    total_error = int(len(err_df))
    error_rate = (total_error / total) if total > 0 else 0.0

    pattern_counts = err_df["error_pattern"].value_counts().to_dict() if total_error > 0 else {}
    pair_counts = err_df["error_pair"].value_counts().head(15).to_dict() if total_error > 0 else {}

    lines = [
        "# Error Analysis Summary",
        "",
        f"- Seed: {SEED}",
        f"- Jumlah data test: {total}",
        f"- Jumlah prediksi salah: {total_error}",
        f"- Error rate: {error_rate:.4f}",
        f"- Kolom teks analisis: `{analysis_text_col}`",
        "",
        "## Pola Error (Heuristic)",
    ]
    if pattern_counts:
        for k, v in pattern_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Tidak ada prediksi salah pada data test.")

    lines.extend(["", "## Top Pasangan Salah Klasifikasi"])
    if pair_counts:
        for k, v in pair_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Tidak ada pasangan salah klasifikasi.")

    lines.extend(
        [
            "",
            "## Catatan",
            "- Pola error bersifat heuristic berbasis aturan sederhana.",
            "- Untuk analisis lanjutan, disarankan inspeksi manual sampel berpengaruh tinggi.",
        ]
    )
    summary_output.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Error analysis samples saved: {sample_output}")
    print(f"[OK] Error analysis summary saved: {summary_output}")


if __name__ == "__main__":
    main()
