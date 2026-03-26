import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_vectorizer_config_grid() -> list[dict]:
    return [
        {"analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "sublinear_tf": True},
        {"analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "sublinear_tf": True},
        {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 2, "max_df": 0.95, "sublinear_tf": True},
    ]


def build_model_grid() -> list[dict]:
    grid = []
    for c in [0.5, 1.0, 2.0]:
        grid.append({"model_name": "tfidf_logreg", "model_type": "logreg", "C": c})
        grid.append({"model_name": "tfidf_linear_svm", "model_type": "linear_svm", "C": c})
    return grid


def build_pipeline(vectorizer_cfg: dict, model_cfg: dict) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=vectorizer_cfg["analyzer"],
        ngram_range=vectorizer_cfg["ngram_range"],
        min_df=vectorizer_cfg["min_df"],
        max_df=vectorizer_cfg["max_df"],
        sublinear_tf=vectorizer_cfg["sublinear_tf"],
    )

    if model_cfg["model_type"] == "logreg":
        clf = LogisticRegression(
            C=model_cfg["C"],
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced",
        )
    elif model_cfg["model_type"] == "linear_svm":
        clf = LinearSVC(
            C=model_cfg["C"],
            class_weight="balanced",
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_cfg['model_type']}")

    return Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 - Baseline models with controlled tuning")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--test", type=str, default="data/test.csv")
    parser.add_argument("--text-col", type=str, default="text_model_input")
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--results-output", type=str, default="outputs/baseline_results.csv")
    parser.add_argument("--report-output", type=str, default="outputs/baseline_comparison.md")
    parser.add_argument("--trials-output", type=str, default="outputs/baseline_trials.csv")
    args = parser.parse_args()

    set_global_seed(SEED)

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    results_output = Path(args.results_output)
    report_output = Path(args.report_output)
    trials_output = Path(args.trials_output)

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required split file not found: {p}")

    for p in [results_output, report_output, trials_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if args.text_col not in df.columns:
            raise KeyError(f"{name} missing text column: {args.text_col}")
        if args.label_col not in df.columns:
            raise KeyError(f"{name} missing label column: {args.label_col}")

    x_train = train_df[args.text_col].fillna("").astype(str)
    y_train = train_df[args.label_col].astype(str)
    x_val = val_df[args.text_col].fillna("").astype(str)
    y_val = val_df[args.label_col].astype(str)
    x_test = test_df[args.text_col].fillna("").astype(str)
    y_test = test_df[args.label_col].astype(str)

    trial_rows: list[dict] = []
    best_by_model: dict[str, dict] = {}

    vectorizer_grid = build_vectorizer_config_grid()
    model_grid = build_model_grid()

    for vcfg in vectorizer_grid:
        for mcfg in model_grid:
            pipe = build_pipeline(vcfg, mcfg)
            pipe.fit(x_train, y_train)
            val_pred = pipe.predict(x_val)
            val_metrics = compute_metrics(y_val, val_pred)

            row = {
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "model": mcfg["model_name"],
                "split": "val",
                "accuracy": round(val_metrics["accuracy"], 6),
                "precision_macro": round(val_metrics["precision_macro"], 6),
                "recall_macro": round(val_metrics["recall_macro"], 6),
                "f1_macro": round(val_metrics["f1_macro"], 6),
                "config_json": json.dumps(
                    {"vectorizer": vcfg, "model": mcfg},
                    ensure_ascii=False,
                    default=str,
                ),
            }
            trial_rows.append(row)

            model_name = mcfg["model_name"]
            if model_name not in best_by_model or row["f1_macro"] > best_by_model[model_name]["val_row"]["f1_macro"]:
                best_by_model[model_name] = {
                    "val_row": row,
                    "vcfg": vcfg,
                    "mcfg": mcfg,
                }

    trial_df = pd.DataFrame(trial_rows).sort_values(["model", "f1_macro"], ascending=[True, False])
    trial_df.to_csv(trials_output, index=False, encoding="utf-8-sig")

    final_rows = []
    for model_name, best in best_by_model.items():
        pipe = build_pipeline(best["vcfg"], best["mcfg"])
        pipe.fit(x_train, y_train)
        test_pred = pipe.predict(x_test)
        test_metrics = compute_metrics(y_test, test_pred)

        final_rows.append(
            {
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "model": model_name,
                "split": "val_best_config",
                "accuracy": best["val_row"]["accuracy"],
                "precision_macro": best["val_row"]["precision_macro"],
                "recall_macro": best["val_row"]["recall_macro"],
                "f1_macro": best["val_row"]["f1_macro"],
                "config_json": best["val_row"]["config_json"],
            }
        )
        final_rows.append(
            {
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "model": model_name,
                "split": "test_best_from_val",
                "accuracy": round(test_metrics["accuracy"], 6),
                "precision_macro": round(test_metrics["precision_macro"], 6),
                "recall_macro": round(test_metrics["recall_macro"], 6),
                "f1_macro": round(test_metrics["f1_macro"], 6),
                "config_json": best["val_row"]["config_json"],
            }
        )

    results_df = pd.DataFrame(final_rows).sort_values(["model", "split"])
    results_df.to_csv(results_output, index=False, encoding="utf-8-sig")

    best_model_on_val = max(
        [r for r in final_rows if r["split"] == "val_best_config"],
        key=lambda x: x["f1_macro"],
    )["model"]
    best_model_on_test = max(
        [r for r in final_rows if r["split"] == "test_best_from_val"],
        key=lambda x: x["f1_macro"],
    )["model"]

    lines = [
        "# Baseline Comparison",
        "",
        f"- Waktu proses: {datetime.now().isoformat()}",
        f"- Seed: {SEED}",
        "- Tuning dilakukan di validation set, test set hanya evaluasi konfigurasi terbaik dari validation.",
        "",
        "## Ringkasan Hasil",
        f"- Model terbaik berdasarkan val f1_macro: `{best_model_on_val}`",
        f"- Model terbaik berdasarkan test f1_macro (best-from-val): `{best_model_on_test}`",
        "",
        "## Metrik Konfigurasi Terbaik",
        "",
        dataframe_to_markdown_simple(results_df),
        "",
        "## Catatan",
        f"- Semua trial validation disimpan di `{trials_output}`",
    ]
    report_output.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Baseline trials saved: {trials_output}")
    print(f"[OK] Baseline results saved: {results_output}")
    print(f"[OK] Baseline report saved: {report_output}")


if __name__ == "__main__":
    main()
