import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


def read_json_first_available(paths: list[Path]) -> dict:
    for p in paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def read_text_first_available(paths: list[Path]) -> str:
    for p in paths:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def format_baseline_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "Data baseline belum tersedia."
    cols = [c for c in ["model", "split", "accuracy", "precision_macro", "recall_macro", "f1_macro"] if c in df.columns]
    sub = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in sub.iterrows():
        lines.append("| " + " | ".join([str(r[c]) for c in cols]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 11 - Generate Bab 4 & Bab 5 draft")
    parser.add_argument("--bab4-output", type=str, default="outputs/bab4_hasil_otomatis.md")
    parser.add_argument("--bab5-output", type=str, default="outputs/bab5_kesimpulan_saran_otomatis.md")
    args = parser.parse_args()

    set_global_seed(SEED)

    bab4_output = Path(args.bab4_output)
    bab5_output = Path(args.bab5_output)
    for p in [bab4_output, bab5_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    audit = read_json_first_available([Path("outputs/data_audit_summary.json")])
    cleaning = read_json_first_available([Path("outputs/cleaning_log.json")])
    preprocessing = read_json_first_available(
        [Path("outputs/preprocessing_log_v4.json"), Path("outputs/preprocessing_log_v3.json"), Path("outputs/preprocessing_log.json")]
    )
    split_summary = read_json_first_available([Path("outputs/split_summary.json")])
    final_metrics = read_json_first_available([Path("outputs/final_metrics.json")])
    best_cfg = read_json_first_available([Path("outputs/best_config.json"), Path("outputs/step7_best_config.json")])
    eda_summary = read_text_first_available([Path("outputs/eda_summary.md")])
    err_summary = read_text_first_available([Path("outputs/error_analysis_summary.md")])
    baseline_df = read_csv_if_exists(Path("outputs/baseline_results.csv"))

    audit_rows = audit.get("audit", {}).get("n_rows", "N/A")
    clean_rows = cleaning.get("rows_after_cleaning", "N/A")
    removed_total = cleaning.get("removed_counts", {}).get("total_removed", "N/A")
    pre_rows = preprocessing.get("rows_processed", "N/A")
    split_sizes = split_summary.get("size", {})

    acc = final_metrics.get("accuracy", "N/A")
    prec = final_metrics.get("precision_macro", "N/A")
    rec = final_metrics.get("recall_macro", "N/A")
    f1 = final_metrics.get("f1_macro", "N/A")

    best_trial_name = best_cfg.get("best_trial_name", best_cfg.get("best_config", {}).get("trial_name", "N/A"))
    best_trial_f1 = best_cfg.get("best_val_f1_macro", "N/A")

    bab4_lines = [
        "# BAB 4 IMPLEMENTASI DAN PENGUJIAN",
        "",
        f"Dokumen ini dihasilkan otomatis pada {datetime.now().isoformat()} dengan seed eksperimen {SEED}.",
        "",
        "## 4.1 Lingkungan Implementasi",
        "- Bahasa pemrograman: Python.",
        "- Framework utama: PyTorch dan Transformers.",
        "- Pipeline dijalankan bertahap dari audit data hingga evaluasi.",
        "",
        "## 4.2 Dataset dan Prapemrosesan",
        f"- Jumlah data mentah hasil audit: {audit_rows} baris.",
        f"- Jumlah data setelah cleaning: {clean_rows} baris.",
        f"- Total data yang dihapus saat cleaning: {removed_total} baris.",
        f"- Jumlah data yang diproses pada tahap preprocessing: {pre_rows} baris.",
        "",
        "## 4.3 Eksplorasi Data (EDA)",
        "Ringkasan EDA otomatis:",
        "",
        eda_summary if eda_summary else "EDA summary belum tersedia.",
        "",
        "## 4.4 Pembagian Data",
        f"- Rasio split utama: train={split_summary.get('split_ratio', {}).get('train', 'N/A')} dan test={split_summary.get('split_ratio', {}).get('test', 'N/A')}.",
        f"- Data train: {split_sizes.get('train', 'N/A')} baris.",
        f"- Data validation (jika ada): {split_sizes.get('val', 0)} baris.",
        f"- Data test: {split_sizes.get('test', 'N/A')} baris.",
        "",
        "## 4.5 Hasil Baseline",
        format_baseline_table(baseline_df),
        "",
        "## 4.6 Model Utama IndoBERT + BiLSTM",
        f"- Konfigurasi terbaik yang tercatat: {best_trial_name}.",
        f"- Nilai validasi terbaik (macro F1) yang tercatat: {best_trial_f1}.",
        "",
        "## 4.7 Evaluasi Final pada Test Set",
        f"- Accuracy: {acc}",
        f"- Precision (macro): {prec}",
        f"- Recall (macro): {rec}",
        f"- F1-score (macro): {f1}",
        "- Confusion matrix dan classification report disimpan di folder outputs.",
        "",
        "## 4.8 Analisis Error",
        err_summary if err_summary else "Ringkasan error analysis belum tersedia.",
    ]
    bab4_output.write_text("\n".join(bab4_lines), encoding="utf-8")

    bab5_lines = [
        "# BAB 5 KESIMPULAN DAN SARAN",
        "",
        f"Dokumen ini dihasilkan otomatis pada {datetime.now().isoformat()} berdasarkan artefak komputasi yang tersedia.",
        "",
        "## 5.1 Kesimpulan",
        "- Pipeline penelitian telah disusun end-to-end secara bertahap dan reproducible.",
        "- Tahapan utama meliputi audit, cleaning, preprocessing, EDA, split data, baseline, model utama, tuning, evaluasi, dan error analysis.",
        f"- Metrik evaluasi final yang tercatat: accuracy={acc}, precision_macro={prec}, recall_macro={rec}, f1_macro={f1}.",
        "",
        "## 5.2 Keterbatasan",
        "- Kualitas hasil sangat dipengaruhi kualitas label dan keberagaman konteks komentar TikTok.",
        "- Class imbalance dapat menurunkan performa macro metrics pada kelas minoritas.",
        "- Resource komputasi memengaruhi kedalaman eksperimen tuning/fine-tuning.",
        "",
        "## 5.3 Saran",
        "- Lakukan validasi ulang sebagian label secara manual untuk mengurangi noise anotasi.",
        "- Tambahkan data dan variasi konteks agar generalisasi model meningkat.",
        "- Uji konfigurasi model lanjutan secara sistematis dengan logging eksperimen lebih detail.",
        "- Pertimbangkan augmentasi atau strategi imbalance lanjutan untuk meningkatkan performa kelas minoritas.",
    ]
    bab5_output.write_text("\n".join(bab5_lines), encoding="utf-8")

    print(f"[OK] Bab 4 draft saved: {bab4_output}")
    print(f"[OK] Bab 5 draft saved: {bab5_output}")


if __name__ == "__main__":
    main()
