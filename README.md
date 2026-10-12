# Earthquake Ground-Motion Directionality with Explainable AI

Predicting how differently two nearby sites are shaken by the same earthquake, and explaining which inputs the model relies on most.

Based on the paper *"Detecting and predicting earthquake ground motion directionality patterns using machine learning tools"* (Poulos & Silva-Lopez), using the NGA-West2 database.

**New feature added in this project:** Explainable AI through permutation feature importance.

---

## What this project does

When an earthquake hits, the ground shakes hardest in one particular direction at each site. Two sites close together often (but not always) share that direction. This project asks: **given two sites that recorded the same earthquake, how far apart (0-90 degrees) are their strongest-shaking directions, and can we predict it?**

Then it adds a second question: **which input variables does the model lean on most to make that prediction?**

Think of it like a recipe. If swapping the salt for sugar ruins the dish, salt matters. If swapping the garnish changes nothing, it does not. Permutation importance does exactly this to each input variable, and reports the result in degrees of extra error.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_build_dataset.py` | Reads the NGA-West2 flatfile and waveforms, finds the strongest-shaking direction at every recording, pairs up sites from the same earthquake, and builds features for each pair |
| 2 | `02_train_models.py` | Trains a distance-only benchmark, ridge regression and a neural network; compares R2 and mean absolute error on unseen earthquakes |
| 3 | `03_feature_importance.py` | Permutation importance (top-5 chart and grouped "family" chart) for ridge and the neural network |
| 4 | `04_demo_explain.py` | Predicts the angle difference for example pairs and prints the most influential variables |

Shared code lives in `common.py` (waveform reading, direction finding, pair features) and `modeling.py` (models, train/test split, readable variable names).

## Setup

```bash
git clone https://github.com/siddeshvs351/earthquake-ground-motion-directionality-ml.git
cd earthquake-ground-motion-directionality-ml
pip install -r requirements.txt
```

Python 3.9+ is recommended.

## Data

The data files are **not included** in this repository because of their size. Place them in a `data/` folder:

```
data/
  <NGA-West2 flatfile>.xlsx      # from the PEER NGA-West2 database
  demo_waveforms.zip             # or a folder of .AT2 waveform files
```

The NGA-West2 flatfile and waveforms are available from the PEER Ground Motion Database (https://ngawest2.berkeley.edu). The first run caches the flatfile so later runs are fast.

## Usage

```bash
python 01_build_dataset.py        # builds outputs/pairs.csv
python 02_train_models.py         # model comparison table + figure
python 03_feature_importance.py   # importance charts and CSVs
python 04_demo_explain.py         # live prediction + explanation
```

Useful options:

```bash
python 01_build_dataset.py --max_dist 5 --period 0   # paper-like settings
python 04_demo_explain.py --model nn                 # explain the neural network
```

### Simulation mode (method test)

```bash
python 02_train_models.py --simulate
python 03_feature_importance.py --simulate
python 04_demo_explain.py --simulate
```

In this mode the angles are invented from distance, then PGA, then Vs30 (plus noise), so the correct ranking is known in advance. It checks that the importance method finds the right answer. It is a **test of the method, not a scientific result**.

## Outputs

Saved in `outputs/`:

- `pairs.csv`, `records_with_direction.csv` - the built dataset
- `metrics_models.csv`, `fig1_true_vs_predicted.png` - model comparison
- `fig_top5_importance.png` - top 5 most important variables (the requested chart)
- `fig_importance_families.png` - importance of whole variable families (distance, PGA, Vs30, ...)
- `importance_ridge.csv`, `importance_neural.csv`, `importance_groups_*.csv`
- `model_bundle.joblib` - trained models

## How the importance is measured

1. Train on some earthquakes, hide others.
2. On the hidden earthquakes, shuffle one variable at a time (20 times) and measure how much the mean absolute error rises.
3. Repeat across 5 different groups of hidden earthquakes and average; the bars show the spread.
4. Rank variables, and also shuffle whole families together so related variables (e.g. average PGA and PGA difference) cannot hide each other.

## Honest notes and limitations

- **The small demo waveforms behave like placeholder data**, not real recordings (peaks do not match the flatfile PGA, and neighbouring values look like random noise). On them every model scores R2 close to 0, so the importances are tiny and smaller than their own spread. That chart is not a physical finding. With real NGA-West2 waveforms, distance and PGA are expected to rank near the top, as in the paper.
- Permutation importance shows what the **model** relies on, not necessarily what the earthquake physics depends on. Strongly related variables can share or hide importance.
- The train/test split is by **earthquake** (the paper split random pairs, which gives flattering scores).
- The demo set pairs sites up to 20 km apart (the paper used 5 km) because 5 km leaves too few pairs.

## Project structure

```
.
├── 01_build_dataset.py
├── 02_train_models.py
├── 03_feature_importance.py
├── 04_demo_explain.py
├── common.py
├── modeling.py
├── requirements.txt
├── README.md
├── data/        # not tracked: flatfile + waveforms
└── outputs/     # generated results
```

## Future work

- Run on the full set of real NGA-West2 waveforms
- Use more sites per earthquake and the paper's 5 km pairing distance
- Compare with other explainability methods (e.g. SHAP)

## Reference

Poulos, A. & Silva-Lopez, R. *Detecting and predicting earthquake ground motion directionality patterns using machine learning tools.*
