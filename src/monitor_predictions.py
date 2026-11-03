"""Lab 12 - Monitoring dashboard from the API prediction log: volume, positive rate,
confidence, latency, live recall/precision per window, input drift, and alerts."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from common import load_config, load_clean_raw, split_raw, psi


def monitor():
    cfg = load_config()
    m = cfg["monitoring"]
    log = pd.read_json(cfg["serving"]["prediction_log"], lines=True)
    if "type" not in log:
        log["type"] = None
    is_fb = log["type"] == "feedback"
    preds = log[~is_fb].drop(columns=["type", "actual"], errors="ignore")
    fb = log.loc[is_fb, ["prediction_id", "actual"]]
    df = preds.merge(fb, on="prediction_id", how="left")
    df = df.reset_index(drop=True)
    df["window"] = df.index // m["window_size"]

    X_ref = split_raw(load_clean_raw())[0]
    windows, alerts = [], []
    for w, g in df.groupby("window"):
        feats = pd.DataFrame(list(g["features"]))
        labelled = g.dropna(subset=["actual"])
        row = {
            "window": int(w), "requests": len(g),
            "positive_rate": round(g["prediction"].mean(), 4),
            "mean_probability": round(g["probability"].mean(), 4),
            "p95_latency_ms": round(g["latency_ms"].quantile(.95), 3),
            "recall": round(recall_score(labelled["actual"], labelled["prediction"], zero_division=0), 4),
            "precision": round(precision_score(labelled["actual"], labelled["prediction"], zero_division=0), 4),
            "max_feature_psi": round(max(psi(X_ref[c], feats[c]) for c in X_ref.columns), 3),
        }
        checks = [
            (row["positive_rate"] > m["max_positive_rate"], f"positive rate {row['positive_rate']:.2f} > {m['max_positive_rate']}"),
            (row["positive_rate"] < m["min_positive_rate"], f"positive rate {row['positive_rate']:.2f} < {m['min_positive_rate']}"),
            (row["p95_latency_ms"] > m["max_latency_ms"], f"p95 latency {row['p95_latency_ms']} ms"),
            (row["recall"] < m["min_window_recall"], f"recall {row['recall']:.2f} < {m['min_window_recall']}"),
            (row["max_feature_psi"] > m["psi_alert"], f"input drift (max PSI {row['max_feature_psi']})"),
        ]
        row["alerts"] = [msg for bad, msg in checks if bad]
        alerts += [f"window {w}: {a}" for a in row["alerts"]]
        windows.append(row)
    wdf = pd.DataFrame(windows)

    # Dashboard
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(wdf["window"], wdf["positive_rate"], "o-", label="positive rate")
    ax[0, 0].plot(wdf["window"], wdf["mean_probability"], "s--", label="mean probability")
    ax[0, 0].axhline(m["max_positive_rate"], c="r", ls=":", lw=1)
    ax[0, 0].set_title("Prediction behaviour")
    ax[0, 1].plot(wdf["window"], wdf["recall"], "o-", label="recall")
    ax[0, 1].plot(wdf["window"], wdf["precision"], "s-", label="precision")
    ax[0, 1].axhline(m["min_window_recall"], c="r", ls=":", lw=1)
    ax[0, 1].set_title("Live performance (from feedback)")
    ax[1, 0].bar(wdf["window"], wdf["max_feature_psi"],
                 color=["#d62728" if v > m["psi_alert"] else "#2ca02c" for v in wdf["max_feature_psi"]])
    ax[1, 0].set_title("Input drift: max feature PSI per window")
    ax[1, 1].plot(wdf["window"], wdf["p95_latency_ms"], "o-")
    ax[1, 1].set_title("p95 latency per request (ms)")
    for a in ax.flat:
        a.set_xlabel(f"window ({m['window_size']} predictions)")
        if a.get_legend_handles_labels()[0]:
            a.legend()
    fig.suptitle(f"Bank Marketing API monitoring - model v{int(df['model_version'].iloc[-1])}")
    fig.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    fig.savefig("artifacts/monitoring_dashboard.png", dpi=110)
    plt.close(fig)

    report = {"total_predictions": len(df), "labelled": int(df["actual"].notna().sum()),
              "status": "ALERT" if alerts else "HEALTHY", "alerts": alerts, "windows": windows}
    with open("artifacts/monitoring_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(wdf.drop(columns="alerts").to_string(index=False))
    print(f"\n[{'WARNING' if alerts else 'SUCCESS'}] Status: {report['status']} ({len(alerts)} alerts)")
    for a in alerts:
        print("  -", a)
    print("[INFO] Saved artifacts/monitoring_dashboard.png and artifacts/monitoring_report.json")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if monitor() else 1)
