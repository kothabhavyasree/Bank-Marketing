"""Lab 10 - Simulate a new month of campaign data whose distribution has shifted
(interest rates fall, employment drops, older customers, more calls per customer)."""
import os
import numpy as np
from common import load_config, load_clean_raw, split_raw


def simulate(n=3000, seed=7):
    cfg = load_config()
    X_train, _, y_train, _ = split_raw(load_clean_raw())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=n, replace=False)
    batch = X_train.iloc[idx].copy().reset_index(drop=True)
    batch["y"] = np.where(y_train.iloc[idx].to_numpy() == 1, "yes", "no")

    # Macro-economic shift (kept inside the Lab 5 schema ranges)
    batch["euribor3m"] = (batch["euribor3m"] * 0.55).clip(0.6, 5.1).round(3)
    batch["emp.var.rate"] = (batch["emp.var.rate"] - 1.2).clip(-3.4, 1.4).round(1)
    batch["nr.employed"] = (batch["nr.employed"] - 90).clip(4963.6, 5228.1).round(1)
    batch["cons.conf.idx"] = (batch["cons.conf.idx"] - 4).clip(-50.8, -26.9).round(1)
    # Customer-mix shift
    older = rng.random(n) < 0.5
    batch.loc[older, "age"] = (batch.loc[older, "age"] + 12).clip(upper=98)
    batch["campaign"] = batch["campaign"] + rng.poisson(2, n)
    batch.loc[rng.random(n) < 0.6, "contact"] = "cellular"

    path = cfg["drift"]["current_batch"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    batch.to_csv(path, sep=";", index=False)
    print(f"[SUCCESS] Drifted batch of {n} labelled rows saved to {path}")


if __name__ == "__main__":
    simulate()
