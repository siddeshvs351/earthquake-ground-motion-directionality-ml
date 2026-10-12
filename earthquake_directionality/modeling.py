"""
Shared modelling helpers: loading the pair table, splitting by earthquake,
building the ridge and neural-network models, and readable variable names.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import OUT

META = ["rsn_i", "rsn_j", "eqid", "delta_theta"]

# ---- readable names (what appears on the charts) --------------------------------
FRIENDLY = {
    "sep_dist_km": "Distance between the two sites",
    "mag": "Earthquake magnitude", "strike": "Fault strike angle", "dip": "Fault dip angle",
    "rake": "Fault rake angle", "hyp_depth": "Depth of earthquake origin",
    "ztor": "Depth to top of fault rupture", "width": "Fault rupture width",
    "length": "Fault rupture length",
    "saz_diff": "Difference in direction to the fault",
    "same_nehrp": "Both sites on same soil class",
}
for _k in range(4):
    FRIENDLY[f"mech_{_k}"] = f"Fault type {_k}"
for _c, _n in {"vs30": "soil stiffness (Vs30)", "pga": "peak ground acceleration (PGA)",
               "pgv": "peak ground velocity (PGV)", "pgd": "peak ground displacement (PGD)",
               "epid": "distance to epicentre", "hypd": "distance to hypocentre",
               "rjb": "Joyner-Boore distance to fault", "rrup": "closest distance to fault"}.items():
    FRIENDLY[f"{_c}_mean"] = f"Average {_n} of the 2 sites"
    FRIENDLY[f"{_c}_absdiff"] = f"Difference in {_n} between sites"

# ---- families of related variables (for the grouped chart) -----------------------
GROUPS = {
    "Distance between the sites": ["sep_dist_km"],
    "PGA (shaking strength)": ["pga_mean", "pga_absdiff"],
    "Vs30 (soil stiffness) and soil class": ["vs30_mean", "vs30_absdiff", "same_nehrp"],
    "PGV / PGD (velocity, displacement)": ["pgv_mean", "pgv_absdiff", "pgd_mean", "pgd_absdiff"],
    "Distance to the fault / epicentre": ["epid_mean", "epid_absdiff", "hypd_mean", "hypd_absdiff",
                                          "rjb_mean", "rjb_absdiff", "rrup_mean", "rrup_absdiff", "saz_diff"],
    "Earthquake size and fault type": ["mag", "strike", "dip", "rake", "hyp_depth", "ztor", "width",
                                       "length", "mech_0", "mech_1", "mech_2", "mech_3"],
}


def load_pairs(cap_per_eq=150, seed=0, simulate=False):
    """Pair table. Pairs per earthquake are capped so one huge earthquake cannot dominate."""
    pairs = pd.read_csv(OUT / "pairs.csv").sample(frac=1, random_state=seed)
    pairs = pairs[pairs.groupby("eqid").cumcount() < cap_per_eq].reset_index(drop=True)
    if simulate:
        # METHOD TEST ONLY: angles are invented from distance, PGA and Vs30 (+ noise), so we
        # KNOW the right answer and can check that the importance chart finds it.
        rng = np.random.RandomState(1)
        pga_r = pairs.pga_mean.rank(pct=True).fillna(0.5)
        vs_r = pairs.vs30_mean.rank(pct=True).fillna(0.5)
        mean = 5 + 2.0 * pairs.sep_dist_km + 20 * (1 - pga_r) + 10 * vs_r
        pairs["delta_theta"] = np.clip(np.abs(mean + 10 * rng.randn(len(pairs))), 0, 90)
        print("*** SIMULATION MODE: angles are simulated (distance > PGA > Vs30 by design). "
              "Method test only. ***")
    feats = [c for c in pairs.columns if c not in META and pairs[c].notna().any()]
    return pairs, feats


def split_by_earthquake(grp, seed=0, test_frac=0.2):
    """Whole earthquakes go either to train or to test (never both), because pairs from one
    earthquake share information and mixing them would give fake-good scores."""
    rng = np.random.RandomState(seed)
    size = pd.Series(grp).value_counts()
    test_ids, n = set(), 0
    for e in rng.permutation(np.unique(grp)):
        if n >= test_frac * len(grp):
            break
        test_ids.add(e); n += size[e]
    is_test = np.array([g in test_ids for g in grp])
    return np.where(~is_test)[0], np.where(is_test)[0]


def make_ridge():
    return GridSearchCV(
        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
        {"ridge__alpha": [1, 10, 100, 300, 1000, 3000]},
        cv=GroupKFold(5), scoring="neg_mean_absolute_error")


def make_nn():
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, early_stopping=True,
                     n_iter_no_change=15, max_iter=500, random_state=0))


def fit_model(model, X, y, grp):
    """Fit; the ridge search needs to know which earthquake each row belongs to."""
    if isinstance(model, GridSearchCV):
        return model.fit(X, y, groups=grp)
    return model.fit(X, y)
