import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 6: Model Registry & Lifecycle")
    print("[INFO] =========================================")

    # V1 is trained and promoted first, then V2 challenges it (champion/challenger).
    steps = [
        ["src/train_registry.py", "v1"],      # Registers version 1
        ["src/automate_lifecycle.py"],        # V1 -> Staging -> Production
        ["src/train_registry.py", "v2"],      # Registers version 2 (challenger)
        ["src/automate_lifecycle.py"],        # V2 vs V1 on recall
        ["src/generate_registry_report.py"],  # JSON deployment artifact
    ]

    for step in steps:
        print(f"\n[INFO] ---> Executing {' '.join(step)}...")
        result = subprocess.run([sys.executable] + step)

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {step[0]} failed with exit code {result.returncode}.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 6 Registry Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()
