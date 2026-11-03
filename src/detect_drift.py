"""Lab 10 - Compare the new batch with the training data (reference):
schema check, PSI + KS per feature, prediction drift, and a retrain decision."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp

from common import load_config, load_clean_raw, split_raw, psi
from validate_data import get_bank_schema


def detect():
    cfg = load_config()
    d = cfg["drift"]
    X_ref, _, _, _ = split_raw(load_clean_raw())
    cur = pd.read_csv(d["current_batch"], sep=";")

    # 1. Schema check (Lab 5 schema minus the dropped 'duration' column)
    schema = get_bank_schema().remove_columns(["duration"])
    try:
        schema.validate(cur, lazy=True)
        schema_ok = True
    except Exception as e:
        schema_ok = False
        print(f"[ERROR] New batch fails schema validation:\n{e}")
    X_cur = cur.drop(columns="y")

    # 2. Feature drift
    rows = []
    for col in X_ref.columns:
        score = psi(X_ref[col], X_cur[col])
        row = {"feature": col, "psi": round(score, 4), "drift": score > d["psi_threshold"]}
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            row["ks_pvalue"] = float(ks_2samp(X_ref[col], X_cur[col]).pvalue)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("psi", ascending=False)
    drift_share = table["drift"].mean()

    # 3. Prediction drift (does the model behave differently on the new data?)
    pipe = joblib.load(cfg["serving"]["inference_pipeline"])
    p_ref = pipe.predict_proba(X_ref.sample(5000, random_state=0))[:, 1]
    p_cur = pipe.predict_proba(X_cur)[:, 1]
    pred_psi = psi(pd.Series(p_ref), pd.Series(p_cur))

    retrain = bool(drift_share > d["drift_share_trigger"] or pred_psi > d["psi_threshold"])
    report = {
        "schema_valid": schema_ok,
        "reference_rows": len(X_ref), "current_rows": len(X_cur),
        "drifted_features": table.loc[table["drift"], "feature"].tolist(),
        "drift_share": round(float(drift_share), 3),
        "prediction_psi": round(pred_psi, 4),
        "mean_probability": {"reference": round(float(p_ref.mean()), 4), "current": round(float(p_cur.mean()), 4)},
        "retrain_required": retrain,
        "features": table.astype(object).where(table.notna(), None).to_dict("records"),
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/drift_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # 4. Plot: PSI per feature + the four most drifted distributions
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(2, 1, 1)
    colors = ["#d62728" if x else "#1f77b4" for x in table["drift"]]
    ax.bar(table["feature"], table["psi"], color=colors)
    ax.axhline(d["psi_threshold"], ls="--", c="k", lw=1, label=f"threshold {d['psi_threshold']}")
    ax.set_title("Population Stability Index per feature (red = drift)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    for i, col in enumerate(table["feature"].head(4)):
        a = fig.add_subplot(2, 4, 5 + i)
        if pd.api.types.is_numeric_dtype(X_ref[col]):
            a.hist(X_ref[col], bins=30, alpha=.5, density=True, label="reference")
            a.hist(X_cur[col], bins=30, alpha=.5, density=True, label="current")
        else:
            pd.DataFrame({"reference": X_ref[col].value_counts(normalize=True),
                          "current": X_cur[col].value_counts(normalize=True)}).plot.bar(ax=a)
        a.set_title(col)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("artifacts/drift_report.png", dpi=110)
    plt.close(fig)

    print(table.head(8).to_string(index=False))
    print(f"\n[INFO] Drifted features: {len(report['drifted_features'])}/{len(table)} "
          f"| prediction PSI = {pred_psi:.3f}")
    print(f"[{'WARNING' if retrain else 'SUCCESS'}] Retrain required: {retrain}")
    print("[INFO] Saved artifacts/drift_report.json and artifacts/drift_report.png")
    return schema_ok


if __name__ == "__main__":
    raise SystemExit(0 if detect() else 1)
