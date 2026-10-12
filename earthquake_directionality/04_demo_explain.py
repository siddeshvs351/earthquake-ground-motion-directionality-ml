"""
STEP 4 - live demonstration: the system predicts the angle difference for two sites AND explains
which input variables matter most to its predictions.

    python 04_demo_explain.py                       # 4 example pairs from unseen earthquakes
    python 04_demo_explain.py --model nn            # explain the neural network instead of ridge
    python 04_demo_explain.py --simulate            # the simulation-mode version
"""
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from common import OUT
from modeling import load_pairs, split_by_earthquake, FRIENDLY

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["ridge", "nn"], default="ridge")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--simulate", action="store_true")
args = ap.parse_args()
P = "sim_" if args.simulate else ""

B = joblib.load(OUT / f"{P}model_bundle.joblib")
tag = "ridge" if args.model == "ridge" else "neural"
imp = pd.read_csv(OUT / f"{P}importance_{tag}.csv")
title = "Ridge regression" if args.model == "ridge" else "Neural network"

pairs, feats = load_pairs(150, 0, args.simulate)
_, te = split_by_earthquake(pairs.eqid.values, 0)
sample = pairs.iloc[te].sample(min(args.n, len(te)), random_state=3)

print(f"=== {title}: what matters most to its predictions (global ranking) ===")
for r in imp.head(5).itertuples():
    print(f"  {r.rank}. {r.name:<58s} scrambling it adds {r.error_increase_deg:+.2f} deg of error")
print()
for r in sample.itertuples():
    row = pairs.loc[[r.Index], B["features"]]
    pred = float(np.clip(B[args.model].predict(row)[0], 0, 90))
    print(f"Sites RSN {int(r.rsn_i)} & RSN {int(r.rsn_j)} | earthquake {int(r.eqid)} | {r.sep_dist_km:.1f} km apart")
    print(f"  Predicted direction difference : {pred:.0f} deg")
    print(f"  {'Simulated value' if args.simulate else 'Measured from waveforms'}{' ' * (9 if args.simulate else 0)}: {r.delta_theta:.0f} deg")
    print("  Most influential variables overall: " + "; ".join(imp.name.head(3)) + "\n")
