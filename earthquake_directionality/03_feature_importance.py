"""
STEP 3 - NEW FEATURE: Explainable AI (permutation feature importance).

Idea in one sentence: take one input variable, shuffle its values so it carries no information,
and see how much worse the predictions get. A big jump in error = the model relies on that
variable. The result is reported in degrees: "scrambling X makes the average error N degrees worse".

To make the answer trustworthy, the test is repeated on 5 groups of earthquakes the model
never saw while training, and averaged.

    python 03_feature_importance.py
    python 03_feature_importance.py --simulate
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from common import OUT
from modeling import *

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--cap_per_eq", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--repeats", type=int, default=20, help="how many times each variable is shuffled")
ap.add_argument("--simulate", action="store_true")
args = ap.parse_args()
P = "sim_" if args.simulate else ""

pairs, FEATS = load_pairs(args.cap_per_eq, args.seed, args.simulate)
X, y, grp = pairs[FEATS], pairs.delta_theta.values, pairs.eqid.values
groups = {g: [c for c in cols if c in FEATS] for g, cols in GROUPS.items()}
rng = np.random.RandomState(args.seed)


def grouped_importance(model, Xte, yte):
    """Same idea, but a whole family of related variables is shuffled together
    (e.g. PGA average + PGA difference), so related variables do not hide each other."""
    base = mean_absolute_error(yte, model.predict(Xte))
    out = {}
    for name, cols in groups.items():
        inc = []
        for _ in range(args.repeats):
            perm = rng.permutation(len(Xte))
            Xp = Xte.copy(); Xp[cols] = Xte[cols].values[perm]
            inc.append(mean_absolute_error(yte, model.predict(Xp)) - base)
        out[name] = np.mean(inc)
    return out


results = {}
for model_name, maker in {"Ridge regression": make_ridge, "Neural network": make_nn}.items():
    per_feat, per_group = [], []
    for k, (a, b) in enumerate(GroupKFold(5).split(X, y, grp), 1):
        m = fit_model(maker(), X.iloc[a], y[a], grp[a])
        r = permutation_importance(m, X.iloc[b], y[b], scoring="neg_mean_absolute_error",
                                   n_repeats=args.repeats, random_state=args.seed)
        per_feat.append(r.importances_mean)               # = increase in error, in degrees
        per_group.append(grouped_importance(m, X.iloc[b], y[b]))
    pf = np.array(per_feat)
    df = pd.DataFrame({"variable": FEATS, "name": [FRIENDLY.get(c, c) for c in FEATS],
                       "error_increase_deg": pf.mean(0), "spread_across_folds": pf.std(0)})
    df = df.sort_values("error_increase_deg", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    pg = pd.DataFrame(per_group)
    gdf = pd.DataFrame({"group": pg.columns, "error_increase_deg": pg.mean().values,
                        "spread_across_folds": pg.std().values}).sort_values(
        "error_increase_deg", ascending=False).reset_index(drop=True)
    results[model_name] = (df, gdf)
    tag = model_name.split()[0].lower()
    df.round(3).to_csv(OUT / f"{P}importance_{tag}.csv", index=False)
    gdf.round(3).to_csv(OUT / f"{P}importance_groups_{tag}.csv", index=False)
    print(f"\n=== {model_name}: top 5 variables (average error gets this many degrees worse if scrambled) ===")
    print(df.head(5)[["rank", "name", "error_increase_deg", "spread_across_folds"]].round(3).to_string(index=False))
    print(f"--- {model_name}: grouped view ---")
    print(gdf.round(3).to_string(index=False))


def barplot(ax, names, vals, errs, title, color):
    pos = np.arange(len(names))[::-1]
    ax.barh(pos, vals, xerr=errs, color=color, capsize=3)
    ax.set_yticks(pos); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Error increase when variable is scrambled (degrees)")
    ax.set_title(title, fontsize=11)


# ---- the requested chart: top five variables --------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for a, (name, (df, _)), c in zip(ax, results.items(), ["#4C72B0", "#DD8452"]):
    t = df.head(5)
    barplot(a, t.name, t.error_increase_deg, t.spread_across_folds, f"{name}: top 5 most important variables", c)
plt.suptitle("Which inputs matter most for predicting the angle difference?", fontsize=13)
plt.tight_layout(); plt.savefig(OUT / f"{P}fig_top5_importance.png", dpi=130); plt.close()

# ---- extra chart: families of variables (distance, PGA, Vs30, ...) -------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
for a, (name, (_, g)), c in zip(ax, results.items(), ["#4C72B0", "#DD8452"]):
    barplot(a, g.group, g.error_increase_deg, g.spread_across_folds, f"{name}: variable families", c)
plt.tight_layout(); plt.savefig(OUT / f"{P}fig_importance_families.png", dpi=130); plt.close()
print(f"\nSaved outputs/{P}fig_top5_importance.png, {P}fig_importance_families.png, {P}importance_*.csv")
