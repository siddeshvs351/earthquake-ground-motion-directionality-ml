"""
STEP 1 - turn raw files into a table where each row is a PAIR of sites that
recorded the same earthquake, with the true answer (angle difference) attached.

    python 01_build_dataset.py
    python 01_build_dataset.py --max_dist 5 --period 0      (paper-like settings)
"""
import argparse
import numpy as np
import pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--flatfile", default=None, help="path to the NGA-West2 .xlsx")
ap.add_argument("--waveforms", default=None, help="folder with .AT2 files (zip is auto-extracted)")
ap.add_argument("--max_dist", type=float, default=5.0,
                help="only pair sites closer than this many km (paper used 5; the small demo set needs ~20)")
ap.add_argument("--period", type=float, default=1.0,
                help="building period in seconds used to measure shaking; 0 = raw ground acceleration")
args = ap.parse_args()

rec = clean_records(load_flatfile(args.flatfile))
waves = find_waveforms(args.waveforms)
print(f"Flatfile records: {len(rec)} | waveform records found: {len(waves)}")
rec = rec[rec.rsn.isin(waves)].copy()

# --- direction of strongest shaking for every record --------------------------
theta, az = [], []
for _, r in rec.iterrows():
    h1, dt = read_at2(waves[int(r.rsn)][0])
    h2, _ = read_at2(waves[int(r.rsn)][1])
    t, _ = strongest_direction(h1, h2, dt, args.period)
    # H2 should be H1 + 90 deg (or - 90 deg); this tells us which way to turn
    turn = (r.h2az - r.h1az) % 360 if pd.notna(r.h1az) and pd.notna(r.h2az) else np.nan
    sign = {90.0: 1, 270.0: -1}.get(turn, np.nan)
    theta.append(t)
    az.append((r.h1az + sign * t) % 180 if pd.notna(sign) else np.nan)
rec["theta_from_h1"] = theta
rec["max_azimuth"] = az          # compass direction of strongest shaking (0..180)
n0 = len(rec)
rec = rec.dropna(subset=["max_azimuth", "lat", "lon"])
print(f"Records with a usable direction: {len(rec)} of {n0}")
rec.to_csv(OUT / "records_with_direction.csv", index=False)

# --- all pairs of sites inside the same earthquake ------------------------------
Li, Ri = [], []
for _, g in rec.groupby("eqid"):
    idx = g.index.to_numpy()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            Li.append(idx[a]); Ri.append(idx[b])
L, R = rec.loc[Li], rec.loc[Ri]
F = pair_features(L, R)
keep = ((F.sep_dist_km <= args.max_dist).values) & (L.sta.values != R.sta.values)
F = F[keep].reset_index(drop=True)
L, R = L[keep].reset_index(drop=True), R[keep].reset_index(drop=True)

pairs = pd.concat([
    pd.DataFrame({"rsn_i": L.rsn.astype(int), "rsn_j": R.rsn.astype(int),
                  "eqid": L.eqid.astype(int),
                  "delta_theta": angle_diff(L.max_azimuth, R.max_azimuth)}), F], axis=1)
pairs.to_csv(OUT / "pairs.csv", index=False)
print(f"Pairs within {args.max_dist} km: {len(pairs)} from {pairs.eqid.nunique()} earthquakes")
print(pairs.delta_theta.describe().round(1).to_string())
print("Saved outputs/pairs.csv")
