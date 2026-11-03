"""Lab 9 - Build the Docker image, run it, and test the live container over HTTP."""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import load_clean_raw, split_raw

IMAGE, NAME, PORT = "bank-marketing-api:latest", "bank-api-lab9", 8000


def sh(*cmd, check=True):
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def call(path, payload=None):
    req = urllib.request.Request(f"http://localhost:{PORT}{path}",
                                 data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    if not shutil.which("docker") or sh("docker", "info", check=False).returncode != 0:
        print("[ERROR] Docker is not installed or not running. Start Docker Desktop and retry.")
        return False
    if not os.path.exists("models/inference_pipeline.pkl"):
        print("[ERROR] models/inference_pipeline.pkl missing - run Lab 7 first.")
        return False

    print("[INFO] Building image (first build takes a few minutes)...")
    build = sh("docker", "build", "-t", IMAGE, ".", check=False)
    if build.returncode != 0:
        print(build.stderr[-2000:])
        return False

    sh("docker", "rm", "-f", NAME, check=False)
    sh("docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:8000", IMAGE)

    try:
        for _ in range(30):                      # wait for uvicorn to start
            try:
                if call("/health")["status"] == "ok":
                    break
            except Exception:
                time.sleep(1)
        else:
            print("[ERROR] Container did not become healthy.")
            print(sh("docker", "logs", NAME, check=False).stdout)
            return False

        customer = split_raw(load_clean_raw())[1].head(1).to_dict("records")[0]
        result = call("/predict", customer)
        info = call("/model-info")
        size = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Size}}").stdout.strip()

        report = {"image": IMAGE, "image_size_mb": round(int(size) / 1e6, 1),
                  "health": "ok", "model_version": info["version"], "sample_prediction": result}
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/docker_deployment_report.json", "w") as f:
            json.dump(report, f, indent=4)
        print(f"[SUCCESS] Container served prediction: {result}")
        print(f"[INFO] Image size: {report['image_size_mb']} MB | report: artifacts/docker_deployment_report.json")
        return True
    finally:
        if "--keep" not in sys.argv:
            sh("docker", "rm", "-f", NAME, check=False)
        else:
            print(f"[INFO] Container left running at http://localhost:{PORT}/docs")


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
