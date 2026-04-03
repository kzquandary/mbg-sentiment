import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def run_cmd(cmd: list[str], step_no: int, step_name: str, logs: list[dict]) -> None:
    print(f"[RUN] Step {step_no} - {step_name}")
    print("      " + " ".join(cmd))
    started = datetime.now().isoformat()
    res = subprocess.run(cmd, check=False)
    ended = datetime.now().isoformat()
    logs.append(
        {
            "step_no": step_no,
            "step_name": step_name,
            "command": cmd,
            "started_at": started,
            "ended_at": ended,
            "return_code": int(res.returncode),
        }
    )
    if res.returncode != 0:
        raise RuntimeError(f"Step {step_no} ({step_name}) gagal dengan exit code {res.returncode}")
    print(f"[OK] Step {step_no} - {step_name}")


def ensure_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")


def prepare_train_sub_and_val(train_csv: str, label_col: str, seed: int = 42) -> None:
    train_path = Path(train_csv)
    ensure_file(str(train_path))
    df = pd.read_csv(train_path)
    if label_col not in df.columns:
        raise KeyError(f"Label column not found in {train_path}: {label_col}")

    train_sub, val_sub = train_test_split(
        df,
        test_size=0.15,
        random_state=seed,
        stratify=df[label_col].astype(str),
    )
    train_sub.to_csv("data/train_sub.csv", index=False, encoding="utf-8-sig")
    val_sub.to_csv("data/val_sub.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] Internal split saved: data/train_sub.csv ({len(train_sub)}), data/val_sub.csv ({len(val_sub)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full research pipeline from Step 1 to Step 11.")
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--until-step", type=int, default=11)
    parser.add_argument("--dataset-source", choices=["clean", "v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-col", type=str, default="Labeling_Sentimen")
    parser.add_argument("--text-col-raw", type=str, default="text")
    parser.add_argument("--text-col-model", type=str, default="text_model_input")
    parser.add_argument("--step7-config-json", type=str, default="src/resources/step7_final_production.json")
    parser.add_argument("--step7-max-trials", type=int, default=1)
    parser.add_argument("--step7-model-name", type=str, default="indobenchmark/indobert-base-p1")
    parser.add_argument("--run-step8", action="store_true", help="Run Step 8 tuning after Step 7.")
    parser.add_argument("--balance-target-count", type=int, default=1500)
    parser.add_argument("--balance-target-by-label-json", type=str, default="")
    parser.add_argument("--run-class-multiplier", action="store_true", help="Tune class multipliers before Step 9.")
    parser.add_argument("--pipeline-log-output", type=str, default="outputs/pipeline_run_log.json")
    args = parser.parse_args()

    if args.from_step < 1 or args.until_step > 11 or args.from_step > args.until_step:
        raise ValueError("Invalid step range. Use 1..11 and from-step <= until-step.")

    dataset_input_map = {
        "clean": "data/dataset_clean.csv",
        "v1": "data/dataset_relabel_mbg_improved.csv",
        "v2": "data/dataset_relabel_mbg_improved_v2_boost.csv",
    }
    split_input = dataset_input_map[args.dataset_source]
    ensure_file(split_input)
    ensure_file(args.step7_config_json)
    cleaned_dataset_path = "data/dataset_pipeline_clean.csv"

    py = sys.executable
    logs: list[dict] = []

    for step in range(args.from_step, args.until_step + 1):
        if step == 1:
            run_cmd(
                [py, "src/01_audit_data.py", "--input", split_input],
                1,
                "Audit Dataset",
                logs,
            )
        elif step == 2:
            run_cmd(
                [
                    py,
                    "src/02_clean_data.py",
                    "--input",
                    split_input,
                    "--text-col",
                    args.text_col_raw,
                    "--label-col",
                    args.label_col,
                    "--output-csv",
                    cleaned_dataset_path,
                    "--output-xlsx",
                    "data/dataset_pipeline_clean.xlsx",
                ],
                2,
                "Cleaning Dataset",
                logs,
            )
        elif step == 3:
            input_for_split = cleaned_dataset_path if Path(cleaned_dataset_path).exists() else split_input
            run_cmd(
                [
                    py,
                    "src/05_split_data.py",
                    "--input",
                    input_for_split,
                    "--text-col",
                    args.text_col_raw,
                    "--label-col",
                    args.label_col,
                    "--train-ratio",
                    "0.7",
                    "--val-ratio",
                    "0",
                    "--test-ratio",
                    "0.3",
                ],
                3,
                "Split Data",
                logs,
            )
            run_cmd(
                [
                    py,
                    "src/03_preprocess_text.py",
                    "--train-input",
                    "data/train.csv",
                    "--test-input",
                    "data/test.csv",
                    "--text-col",
                    args.text_col_raw,
                ],
                3,
                "Preprocess Train/Test",
                logs,
            )
        elif step == 4:
            run_cmd(
                [py, "src/04_eda.py", "--input", "data/preprocessed_dataset.csv", "--label-col", args.label_col, "--text-col", "text_eda_input"],
                4,
                "EDA",
                logs,
            )
        elif step == 5:
            # Step 5 mandatory output already produced by split in step 3.
            print("[OK] Step 5 - Split artifact already generated in Step 3 (data/train.csv, data/test.csv).")
            logs.append(
                {
                    "step_no": 5,
                    "step_name": "Split Data Artifact Check",
                    "command": [],
                    "started_at": datetime.now().isoformat(),
                    "ended_at": datetime.now().isoformat(),
                    "return_code": 0,
                }
            )
        elif step == 6:
            run_cmd(
                [
                    py,
                    "src/06_baseline_models.py",
                    "--train",
                    "data/train.csv",
                    "--test",
                    "data/test.csv",
                    "--text-col",
                    args.text_col_model,
                    "--label-col",
                    args.label_col,
                ],
                6,
                "Baseline Models",
                logs,
            )
        elif step == 7:
            prepare_train_sub_and_val("data/train.csv", args.label_col, seed=args.seed)
            balance_cmd = [
                py,
                "src/05b_balance_train.py",
                "--input",
                "data/train_sub.csv",
                "--label-col",
                args.label_col,
                "--output",
                "data/train_sub_balanced.csv",
                "--log-output",
                "outputs/train_sub_balance_log.json",
            ]
            if args.balance_target_by_label_json:
                ensure_file(args.balance_target_by_label_json)
                balance_cmd.extend(["--target-by-label-json", args.balance_target_by_label_json])
            else:
                balance_cmd.extend(["--target-mode", "fixed", "--target-count", str(args.balance_target_count)])
            run_cmd(balance_cmd, 7, "Balance Train Subset", logs)

            run_cmd(
                [
                    py,
                    "src/07_indobert_bilstm.py",
                    "--train",
                    "data/train_sub_balanced.csv",
                    "--val",
                    "data/val_sub.csv",
                    "--model-name",
                    args.step7_model_name,
                    "--trial-configs-json",
                    args.step7_config_json,
                    "--max-trials",
                    str(args.step7_max_trials),
                ],
                7,
                "Train IndoBERT+BiLSTM",
                logs,
            )
        elif step == 8:
            if args.run_step8:
                run_cmd(
                    [
                        py,
                        "src/08_tuning.py",
                        "--train",
                        "data/train_sub_balanced.csv",
                        "--model-name",
                        args.step7_model_name,
                        "--internal-val-ratio",
                        "0.15",
                    ],
                    8,
                    "Hyperparameter Tuning",
                    logs,
                )
            else:
                print("[SKIP] Step 8 - gunakan --run-step8 jika ingin tuning.")
                logs.append(
                    {
                        "step_no": 8,
                        "step_name": "Hyperparameter Tuning (Skipped)",
                        "command": [],
                        "started_at": datetime.now().isoformat(),
                        "ended_at": datetime.now().isoformat(),
                        "return_code": 0,
                    }
                )
        elif step == 9:
            if args.run_class_multiplier:
                run_cmd(
                    [
                        py,
                        "src/09b_tune_class_multipliers.py",
                        "--val",
                        "data/val_sub.csv",
                        "--model-path",
                        "models/best_indobert_bilstm.pt",
                        "--output",
                        "outputs/best_class_multipliers.json",
                        "--summary-output",
                        "outputs/class_multiplier_tuning_summary.json",
                    ],
                    9,
                    "Tune Class Multipliers",
                    logs,
                )
                eval_cmd = [
                    py,
                    "src/09_evaluate.py",
                    "--test",
                    "data/test.csv",
                    "--model-path",
                    "models/best_indobert_bilstm.pt",
                    "--class-multiplier-json",
                    "outputs/best_class_multipliers.json",
                ]
            else:
                eval_cmd = [
                    py,
                    "src/09_evaluate.py",
                    "--test",
                    "data/test.csv",
                    "--model-path",
                    "models/best_indobert_bilstm.pt",
                ]
            run_cmd(eval_cmd, 9, "Final Evaluation", logs)
        elif step == 10:
            run_cmd(
                [py, "src/10_error_analysis.py", "--pred-input", "outputs/test_predictions.csv"],
                10,
                "Error Analysis",
                logs,
            )
        elif step == 11:
            run_cmd(
                [py, "src/11_generate_report.py"],
                11,
                "Generate Bab 4 & 5 Draft",
                logs,
            )

    log_path = Path(args.pipeline_log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "dataset_source": args.dataset_source,
                "split_input": split_input,
                "cleaned_dataset_path": cleaned_dataset_path,
                "from_step": args.from_step,
                "until_step": args.until_step,
                "step7_config_json": args.step7_config_json,
                "step7_max_trials": args.step7_max_trials,
                "run_step8": args.run_step8,
                "run_class_multiplier": args.run_class_multiplier,
                "logs": logs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Pipeline log saved: {log_path}")


if __name__ == "__main__":
    main()
