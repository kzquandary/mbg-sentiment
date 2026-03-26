# MBG Sentiment Research Pipeline

Pipeline penelitian analisis sentimen komentar TikTok untuk program MBG (IndoBERT + BiLSTM), disusun step-by-step dan reproducible.

## Struktur

```text
project/
  data/
  models/
  notebooks/
  outputs/
    figures/
  reports/
  src/
```

## Dataset

Gunakan nama dataset utama:
- `data/dataset.xlsx`

## Setup

```bash
pip install -r requirements.txt
```

## Menjalankan Pipeline (Script)

```bash
python run_pipeline.py --from-step 1 --until-step 11
```

Atau per step:

```bash
python src/01_audit_data.py --input data/dataset.xlsx
python src/02_clean_data.py --input data/dataset.xlsx
python src/03_preprocess_text.py --input data/cleaned_dataset.csv
python src/04_eda.py --input data/preprocessed_dataset.csv
python src/05_split_data.py --input data/preprocessed_dataset.csv
python src/06_baseline_models.py
python src/07_indobert_bilstm.py
python src/08_tuning.py
python src/09_evaluate.py
python src/10_error_analysis.py
python src/11_generate_report.py
```

## Notebook Resmi

Hanya ada 2 notebook:
- `notebooks/local_full_pipeline.ipynb` -> untuk laptop lokal
- `notebooks/collab_full_pipeline.ipynb` -> untuk Google Colab

Kedua notebook menampilkan tabel/grafik langsung inline, sehingga bisa dipakai walau tanpa mengandalkan file output terpisah.

## Catatan Performa

Performa saat ini sekitar macro F1 ~50% (baseline model utama terbaru), dan masih bisa ditingkatkan lewat audit label dan tuning lanjutan.
