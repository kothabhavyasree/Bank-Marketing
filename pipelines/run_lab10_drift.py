import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 10: Data Drift Detection")
    print("[INFO] =========================================")

    steps = [
        ['src/simulate_drift.py'],
        ['src/detect_drift.py'],
    ]

    for step in steps:
        print(f"\n[INFO] ---> Executing {' '.join(step)}...")
        result = subprocess.run([sys.executable] + step)

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {' '.join(step)} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 10 Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
