"""
Shared helpers: reading waveforms, finding the direction of strongest shaking,
building the "pair of sites" table and its features.
"""
import re
import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Waveforms  ->  direction of strongest shaking
# ----------------------------------------------------------------------------
def read_at2(path):
    """Read a PEER .AT2 acceleration file. Returns (acceleration in g, dt in s)."""
    lines = Path(path).read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if "NPTS" in l.upper())
    m = re.search(r"NPTS\s*=\s*(\d+).*?DT\s*=\s*([0-9.eE+-]+)", lines[k], re.I)
    npts, dt = int(m.group(1)), float(m.group(2))
    vals = np.array(" ".join(lines[k + 1:]).split(), dtype=float)
    return vals[:npts], dt


def sdof_displacement(acc, dt, period, damping=0.05):
    """Response of a simple 'building on a stick' (one-degree-of-freedom system)
    to ground acceleration. Used to measure shaking at one building period."""
    w = 2 * np.pi / period
    A = np.array([[0.0, 1.0], [-w ** 2, -2 * damping * w]])
    B = np.array([[0.0], [-1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt, method="foh")
    num, den = signal.ss2tf(Ad, Bd, Cd, Dd)
    return signal.lfilter(num[0], den, acc)


def strongest_direction(h1, h2, dt, period=1.0):
    """Spin the two horizontal recordings through 0..179 degrees and return the
    angle (measured from H1 toward H2) where the peak shaking is largest
    (the 'RotD100' direction). period=0 uses raw ground acceleration (PGA)."""
    n = min(len(h1), len(h2))
    h1, h2 = h1[:n], h2[:n]
    if period and period > 0:
        h1 = sdof_displacement(h1, dt, period)
        h2 = sdof_displacement(h2, dt, period)
    th = np.deg2rad(np.arange(0, 180, 1.0))
    peak = np.abs(np.outer(np.cos(th), h1) + np.outer(np.sin(th), h2)).max(axis=1)
    k = int(peak.argmax())
    return float(k), float(peak[k])


def angle_diff(a, b):
    """Smallest difference between two directions (a direction and its opposite
    are the same line), in the range 0..90 degrees."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


# ----------------------------------------------------------------------------
# 2. Flatfile -> clean table of records
# ----------------------------------------------------------------------------
COLS = {
    "rsn": "Record Sequence Number", "eqid": "EQID", "sta": "Station Sequence Number",
    "mag": "Earthquake Magnitude", "strike": "Strike (deg)", "dip": "Dip (deg)",
    "rake": "Rake Angle (deg)", "mech": "Mechanism Based on Rake Angle",
    "hyp_depth": "Hypocenter Depth (km)", "ztor": "Depth to Top Of Fault Rupture Model",
    "width": "Fault Rupture Width (km)", "length": "Fault Rupture Length for Calculation of Ry (km)",
    "vs30": "Vs30 (m/s) selected for analysis", "pga": "PGA (g)", "pgv": "PGV (cm/sec)",
    "pgd": "PGD (cm)", "epid": "EpiD (km)", "hypd": "HypD (km)",
    "rjb": "Joyner-Boore Dist. (km)", "rrup": "ClstD (km)",
    "saz": "Source to Site Azimuth (deg)", "nehrp": "Preferred NEHRP Based on Vs30",
    "lat": "Station Latitude", "lon": "Station Longitude",
    "h1az": "H1 azimth (degrees)", "h2az": "H2 azimith (degrees)",
}


def load_flatfile(path=None):
    """Load the NGA-West2 flatfile (slow the first time, cached afterwards)."""
    path = Path(path) if path else next(iter(sorted(DATA.glob("*.xlsx"))))
    cache = DATA / "flatfile_cache.pkl"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pd.read_pickle(cache)
    print(f"Reading {path.name} (takes 1-2 minutes the first time)...")
    raw = pd.read_excel(path)
    raw.to_pickle(cache)
    return raw


def clean_records(raw):
    df = pd.DataFrame({k: raw[v] for k, v in COLS.items()})
    num = [c for c in df.columns if c not in ("nehrp",)]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[num] = df[num].mask(df[num] == -999)           # -999 means "unknown"
    return df


def find_waveforms(wave_dir=None):
    """Return dict {rsn: (H1 path, H2 path)}. Unzips demo_waveforms.zip if needed."""
    wave_dir = Path(wave_dir) if wave_dir else DATA / "waveforms"
    if not list(wave_dir.glob("*.AT2")):
        for z in DATA.glob("*waveforms*.zip"):
            wave_dir.mkdir(exist_ok=True)
            zipfile.ZipFile(z).extractall(wave_dir)
    out = {}
    for f in sorted(glob.glob(str(wave_dir / "**" / "*.AT2"), recursive=True)):
        m = re.search(r"RSN(\d+)_.*-H([12])\.AT2$", f, re.I)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
    return {k: (v[1], v[2]) for k, v in out.items() if 1 in v and 2 in v}


# ----------------------------------------------------------------------------
# 3. Pair of sites -> features
# ----------------------------------------------------------------------------
LOG_COLS = ["vs30", "pga", "pgv", "pgd", "epid", "hypd", "rjb", "rrup"]
EVENT_COLS = ["mag", "strike", "dip", "rake", "hyp_depth", "ztor", "width", "length"]
MECH_LEVELS = [0, 1, 2, 3]


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pair_features(L, R):
    """L and R are equally long tables of records (site i and site j of the same
    earthquake). Every feature is symmetric: swapping the sites changes nothing,
    because the answer (angle difference) is also symmetric."""
    L, R = L.reset_index(drop=True), R.reset_index(drop=True)
    f = pd.DataFrame(index=L.index)
    f["sep_dist_km"] = haversine_km(L.lat, L.lon, R.lat, R.lon)
    for c in EVENT_COLS:
        f[c] = L[c]
    for k in MECH_LEVELS:
        f[f"mech_{k}"] = (L["mech"] == k).astype(float)
    for c in LOG_COLS:
        a, b = np.log10(L[c].clip(lower=0.01)), np.log10(R[c].clip(lower=0.01))
        f[f"{c}_mean"] = (a + b) / 2
        f[f"{c}_absdiff"] = (a - b).abs()
    d = (L.saz - R.saz).abs() % 360
    f["saz_diff"] = np.minimum(d, 360 - d)
    f["same_nehrp"] = (L.nehrp.astype(str) == R.nehrp.astype(str)).astype(float)
    return f


FEATURE_NOTE = (
    "Features: separation distance, earthquake properties (magnitude, fault angles, depth), "
    "and for the two sites the mean and the difference of soil stiffness (Vs30), shaking "
    "strength (PGA/PGV/PGD) and distances to the fault."
)
