"""
STEP 2 - train the three models of the paper (benchmark, ridge regression, neural network)
         and compare them on earthquakes the models have never seen.

    python 02_train_models.py
    python 02_train_models.py --simulate      (method test, see README)
"""
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from common import OUT
from modeling import *

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--cap_per_eq", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--simulate", action="store_true")
args = ap.parse_args()
P = "sim_" if args.simulate else ""

pairs, FEATS = load_pairs(args.cap_per_eq, args.seed, args.simulate)
X, y, grp = pairs[FEATS], pairs.delta_theta.values, pairs.eqid.values
tr, te = split_by_earthquake(grp, args.seed)
print(f"{len(pairs)} pairs, {pairs.eqid.nunique()} earthquakes, {len(FEATS)} variables")
print(f"train {len(tr)} pairs | test {len(te)} pairs from {len(np.unique(grp[te]))} unseen earthquakes")

bench = LinearRegression().fit(X.iloc[tr][["sep_dist_km"]].fillna(0), y[tr])
ridge = fit_model(make_ridge(), X.iloc[tr], y[tr], grp[tr])
nn = fit_model(make_nn(), X.iloc[tr], y[tr], grp[tr])
preds = {
    "Benchmark (distance only)": bench.predict(X.iloc[te][["sep_dist_km"]].fillna(0)),
    "Ridge regression": ridge.predict(X.iloc[te]),
    "Neural network": nn.predict(X.iloc[te]),
}
preds = {k: np.clip(v, 0, 90) for k, v in preds.items()}
rows = [[k, r2_score(y[te], p), mean_absolute_error(y[te], p)] for k, p in preds.items()]
rows.append(["Always guess 45 deg (reference)", r2_score(y[te], np.full(len(te), 45.0)),
             mean_absolute_error(y[te], np.full(len(te), 45.0))])
table = pd.DataFrame(rows, columns=["Model", "R2 (test)", "Mean abs error, deg (test)"]).round(3)
print("\n=== Model comparison (same layout as Table 1 of the paper) ===")
print(table.to_string(index=False))
table.to_csv(OUT / f"{P}metrics_models.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, (name, p) in zip(ax, preds.items()):
    a.scatter(y[te], p, s=8, alpha=.5); a.plot([0, 90], [0, 90], "k--", lw=1)
    a.set(title=f"{name}\nR2={r2_score(y[te], p):.3f}", xlabel="True Δθ (deg)", ylabel="Predicted Δθ (deg)")
plt.tight_layout(); plt.savefig(OUT / f"{P}fig1_true_vs_predicted.png", dpi=130); plt.close()

joblib.dump(dict(features=FEATS, ridge=ridge, nn=nn), OUT / f"{P}model_bundle.joblib")
print(f"\nSaved outputs/{P}metrics_models.csv, {P}fig1_true_vs_predicted.png, {P}model_bundle.joblib")
