import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.common import SEED, backup_if_exists, ensure_dir, set_global_seed


def build_tuning_configs(model_name: str) -> list[dict]:
    # Kombinasi terbatas agar realistis untuk resource laptop
    return [
        {
            "trial_name": "frozen_bs8_len96_h128_do03_lr1e3_adamw",
            "model_name": model_name,
            "max_len": 96,
            "batch_size": 8,
            "hidden_size": 128,
            "dropout": 0.3,
            "lr": 1e-3,
            "epochs": 4,
            "freeze_bert": True,
            "unfreeze_last_n": 0,
            "optimizer": "adamw",
            "patience": 2,
        },
        {
            "trial_name": "frozen_bs6_len128_h192_do04_lr8e4_adamw",
            "model_name": model_name,
            "max_len": 128,
            "batch_size": 6,
            "hidden_size": 192,
            "dropout": 0.4,
            "lr": 8e-4,
            "epochs": 4,
            "freeze_bert": True,
            "unfreeze_last_n": 0,
            "optimizer": "adamw",
            "patience": 2,
        },
        {
            "trial_name": "frozen_bs8_len128_h128_do03_lr7e4_adamw",
            "model_name": model_name,
            "max_len": 128,
            "batch_size": 8,
            "hidden_size": 128,
            "dropout": 0.3,
            "lr": 7e-4,
            "epochs": 4,
            "freeze_bert": True,
            "unfreeze_last_n": 0,
            "optimizer": "adamw",
            "patience": 2,
        },
        {
            "trial_name": "finetune1_bs4_len96_h128_do03_lr2e5_adamw",
            "model_name": model_name,
            "max_len": 96,
            "batch_size": 4,
            "hidden_size": 128,
            "dropout": 0.3,
            "lr": 2e-5,
            "epochs": 3,
            "freeze_bert": False,
            "unfreeze_last_n": 1,
            "optimizer": "adamw",
            "patience": 1,
        },
        {
            "trial_name": "finetune1_bs4_len128_h128_do02_lr15e5_adamw",
            "model_name": model_name,
            "max_len": 128,
            "batch_size": 4,
            "hidden_size": 128,
            "dropout": 0.2,
            "lr": 1.5e-5,
            "epochs": 3,
            "freeze_bert": False,
            "unfreeze_last_n": 1,
            "optimizer": "adamw",
            "patience": 1,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8 - Hyperparameter tuning")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--val", type=str, default="data/val.csv")
    parser.add_argument("--model-name", type=str, default="indobenchmark/indobert-base-p1")
    parser.add_argument("--target-f1", type=float, default=None)
    parser.add_argument("--config-json-output", type=str, default="outputs/tuning_configs.json")
    parser.add_argument("--trial-output", type=str, default="outputs/step7_trials.csv")
    parser.add_argument("--experiment-output", type=str, default="outputs/experiment_results.csv")
    parser.add_argument("--best-config-output", type=str, default="outputs/best_config.json")
    args = parser.parse_args()

    set_global_seed(SEED)

    config_json_output = Path(args.config_json_output)
    trial_output = Path(args.trial_output)
    experiment_output = Path(args.experiment_output)
    best_config_output = Path(args.best_config_output)

    for p in [config_json_output, trial_output, experiment_output, best_config_output]:
        ensure_dir(p.parent)
        backup_if_exists(p)

    configs = build_tuning_configs(args.model_name)
    config_json_output.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd = [
        sys.executable,
        "src/07_indobert_bilstm.py",
        "--train",
        args.train,
        "--val",
        args.val,
        "--model-name",
        args.model_name,
        "--max-trials",
        str(len(configs)),
        "--trial-configs-json",
        str(config_json_output),
        "--best-model-output",
        "models/best_indobert_bilstm.pt",
        "--history-output",
        "outputs/training_history.csv",
        "--trials-output",
        str(trial_output),
        "--arch-output",
        "outputs/model_architecture.md",
        "--best-config-output",
        "outputs/step7_best_config.json",
    ]
    if args.target_f1 is not None:
        cmd.extend(["--target-f1", str(args.target_f1)])

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step 7 tuning call failed with exit code {result.returncode}")

    if not trial_output.exists():
        raise FileNotFoundError(f"Trial output not found: {trial_output}")

    trials_df = pd.read_csv(trial_output)
    trials_df = trials_df.sort_values("val_f1_macro", ascending=False).reset_index(drop=True)

    experiment_df = trials_df.copy()
    experiment_df["timestamp_tuning"] = datetime.now().isoformat()
    experiment_df["seed"] = SEED
    experiment_df.to_csv(experiment_output, index=False, encoding="utf-8-sig")

    best_row = experiment_df.iloc[0].to_dict()
    best_payload = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "target_f1_macro": args.target_f1,
        "best_trial_name": best_row.get("trial_name"),
        "best_val_f1_macro": float(best_row.get("val_f1_macro")),
        "best_config_json": best_row.get("config_json"),
        "target_achieved": (bool(float(best_row.get("val_f1_macro")) >= args.target_f1) if args.target_f1 is not None else None),
    }
    best_config_output.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Tuning config saved: {config_json_output}")
    print(f"[OK] Trial summary saved: {trial_output}")
    print(f"[OK] Experiment results saved: {experiment_output}")
    print(f"[OK] Best config saved: {best_config_output}")


if __name__ == "__main__":
    main()
