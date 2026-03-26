import argparse
import subprocess
import sys
from pathlib import Path


STEP_SCRIPTS = {
    1: "src/01_audit_data.py",
    2: "src/02_clean_data.py",
    3: "src/03_preprocess_text.py",
    4: "src/04_eda.py",
    5: "src/05_split_data.py",
    6: "src/06_baseline_models.py",
    7: "src/07_indobert_bilstm.py",
    8: "src/08_tuning.py",
    9: "src/09_evaluate.py",
    10: "src/10_error_analysis.py",
    11: "src/11_generate_report.py",
}


def run_step(step: int) -> int:
    script = Path(STEP_SCRIPTS[step])
    if not script.exists():
        print(f"[SKIP] Step {step}: script belum ada -> {script}")
        return 0
    print(f"[RUN] Step {step}: {script}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"[FAIL] Step {step}: exit code {result.returncode}")
        return result.returncode
    print(f"[OK] Step {step} selesai")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research pipeline sequentially.")
    parser.add_argument("--from-step", type=int, default=1, help="Step awal")
    parser.add_argument("--until-step", type=int, default=1, help="Step akhir")
    args = parser.parse_args()

    if args.from_step < 1 or args.until_step > 11 or args.from_step > args.until_step:
        raise ValueError("Rentang step tidak valid. Gunakan 1..11 dan from-step <= until-step.")

    for step in range(args.from_step, args.until_step + 1):
        code = run_step(step)
        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()
