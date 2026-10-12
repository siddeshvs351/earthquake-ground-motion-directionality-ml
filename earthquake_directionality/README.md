# Predicting earthquake ground-motion directionality, with Explainable AI

Based on the paper *"Detecting and predicting earthquake ground motion directionality patterns using
machine learning tools"* (Poulos & Silva-Lopez).
**New feature added: Explainable AI - permutation feature importance.**

> "Our system not only predicts the angular difference between two sites, but also explains which
> input variables matter most to its predictions."

## 1. What the assignment asks you to do

| # | Task (from the paper) | Where |
|---|---|---|
| 1 | Clean the NGA-West2 flatfile (keep reliable, meaningful columns; -999 = unknown) | `common.py` -> `clean_records` |
| 2 | From the waveforms, find the direction of strongest shaking at each recording | `common.py` -> `strongest_direction` |
| 3 | Pair sites that recorded the same earthquake; target = difference of directions (0-90 deg) | `01_build_dataset.py` |
| 4 | Build features per pair (distance, soil, shaking strength, earthquake properties) | `common.py` -> `pair_features` |
| 5 | Train benchmark, ridge regression, neural network; compare R2 and mean absolute error | `02_train_models.py` |
| 6 | Discuss results and limits | section 4 below |
| 7 | **New feature**: explain which variables matter most (permutation importance, top-5 bar chart) | `03_feature_importance.py`, `04_demo_explain.py` |

## 2. How to run

```
pip install -r requirements.txt
# put the flatfile (.xlsx) and demo_waveforms.zip in the data/ folder
python 01_build_dataset.py
python 02_train_models.py
python 03_feature_importance.py
python 04_demo_explain.py
python 02_train_models.py --simulate       # method test (section 4)
python 03_feature_importance.py --simulate
python 04_demo_explain.py --simulate
```

## 3. The new feature in plain words

Permutation importance asks one question per variable: **"If I scramble this variable so it carries
no information, how much worse do the predictions get?"**
Think of a recipe: swap the salt for sugar and the dish is ruined (salt matters); swap the garnish and
nobody notices (garnish does not). The answer is in degrees: "scrambling the distance makes the
average error 6.7 degrees worse".

How it is done here:
1. Train the model on some earthquakes, keep other earthquakes hidden.
2. On the hidden earthquakes, shuffle one variable at a time (20 times) and measure the increase in
   mean absolute error.
3. Repeat on 5 different groups of hidden earthquakes and average (the bars show the spread).
4. Rank the variables; show the top 5 as a bar chart, for ridge regression and for the neural network.
5. Extra: shuffle whole families together (distance, PGA, Vs30, ...). Related variables such as
   "average PGA" and "PGA difference" can otherwise hide each other.

Outputs: `fig_top5_importance.png` (the requested chart), `fig_importance_families.png`,
`importance_ridge.csv`, `importance_neural.csv`, `importance_groups_*.csv`.

## 4. IMPORTANT - honest reading of the results

* **The demo waveforms behave like placeholder data, not real recordings** (peaks do not match the
  flatfile PGA, all files have 1,500 samples, neighbouring values are unrelated like random noise).
  On them every model scores R2 about 0, so all importances are tiny (about 0.02 deg) and smaller than
  their own spread. The top-5 chart on the demo data is therefore NOT a physical finding. With
  real NGA-West2 waveforms the same code should rank distance and PGA near the top, as the paper reports.
* **`--simulate`** is a method test: angles are invented from distance, then PGA, then Vs30, plus noise,
  so the right ranking is known. Ridge recovers exactly distance > PGA > Vs30; the neural network finds
  distance first but is noisier on the rest (small data). This shows the feature works; state in your
  report that it is a test, not a result.
* Permutation importance explains what the *model* relies on, not necessarily what the earthquake
  physics depends on. Strongly related variables can share or hide importance (hence the family chart).
* Train/test split is by **earthquake** (the paper split random pairs, which flatters scores).
* The demo set pairs sites up to 20 km apart (paper: 5 km) because 5 km leaves too few pairs.

## 5. Suggested demonstration (5-7 minutes)

1. Idea: Figure 1 of the paper (nearby sites share a strongest-shaking direction).
2. `01_build_dataset.py`: show pair count and `outputs/pairs.csv`.
3. `02_train_models.py`: show the three-model table and `fig1_true_vs_predicted.png`. Explain honestly why the demo data gives R2 about 0.
4. New feature: `03_feature_importance.py --simulate`, show `sim_fig_top5_importance.png`
   ("scramble distance and the error jumps by about 7 degrees; scramble the others and it barely moves").
5. `04_demo_explain.py --simulate`: prediction plus the most influential variables.
6. Conclusion and future work: real waveforms, more sites per earthquake.
