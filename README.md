# MDS6 – Cognitive Computing  
## Asynchronous Teaching Materials

This repository contains the **asynchronous teaching materials and coding notebooks** for the course **MDS6 – Cognitive Computing**.

The materials complement the in-person lectures and focus on **hands-on implementation and deeper exploration of core machine learning concepts**.

---

## Course Context

During the in-person sessions of the course, we cover the **theoretical foundations of machine learning**, including:

- linear and logistic regression
- decision trees and ensemble methods
- evaluation methodology
- experimental design
- model interpretation

The asynchronous materials in this repository extend these topics with **guided coding examples and practical workflows**.

The goal is to help students understand how machine learning methods are applied in realistic scenarios.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_async_session.ipynb
│   ├── 02_async_session.ipynb
├── utils/
│   ├── __init__.py
│   ├── model_curves.py
├── data/   # generated outputs (see workflow below)
└── README.md

``` 

### Notebooks

The notebooks are organized by asynchronous session.

#### Notebook workflow (run order)

- **`notebooks/01_async_session.ipynb`**  
  Builds a synthetic, ML-ready dataset end-to-end:
  - feature engineering + label construction (with leakage-safe time windows),
  - basic preprocessing checks,
  - correlation/redundancy discussion,
  - train/validation split and **scaling fit on train only** (leakage avoidance),
  - exports ready-to-use artifacts into `data/` (e.g. `X_train*.csv`, `y_train*.csv`, `feature_columns.csv`, optional `standard_scaler.joblib`).

- **`notebooks/02_async_session.ipynb`**  
  Loads the **pre-split, pre-scaled** artifacts from `data/` and performs:
  - baseline models, cross-validation, model comparison,
  - hyperparameter tuning (GridSearchCV),
  - feature selection,
  - final evaluation,
  - feature importance (native RF + SHAP showcase).

--- 

## Learning Objectives

The asynchronous materials focus on practical aspects that often occur in real-world machine learning projects:

- constructing datasets from raw operational data
- engineering meaningful features
- avoiding data leakage

The emphasis is on **understanding the reasoning behind each step**, not only running code.

--- 

## Usage

Students can explore the materials in several ways:

- open notebooks directly in **Jupyter**
- run them locally using **Python**
- adapt the examples for their **course case studies**

All notebooks are designed to run **top-to-bottom without external data dependencies**, using synthetic or publicly available datasets.

### Quickstart (local run)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open and run, in order:

1. `notebooks/01_async_session.ipynb` (generates `data/` artifacts)
2. `notebooks/02_async_session.ipynb` (trains/evaluates models using those artifacts)

### Optional dependency: SHAP

`02_async_session.ipynb` includes a SHAP demo cell. If your environment does not have SHAP installed, install it with:

```bash
pip install shap
```

### Note about the `data/` folder (generated outputs)

The notebooks write intermediate and final artifacts into `data/` (CSV and optional `.joblib`). For teaching repos, it is usually best to **treat these as generated outputs**:

- keep them locally for running notebooks,
- but avoid committing large regenerated files unless you want a “zero-run” demo snapshot.

--- 

## License

This repository contains teaching materials for the course  
*MDS6 – Cognitive Computing*.

The materials are licensed under the  
**Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)**.

You may:

- share the material
- adapt the material
- reuse it for educational purposes

Under the conditions that:

- proper attribution is given
- the material is not used for commercial purposes

Full license text:  
https://creativecommons.org/licenses/by-nc/4.0/

---

## Author

Dr. Min Ye  
Founder & CEO, AInvone GmbH

Course: **MDS6 – Cognitive Computing**

---

## Acknowledgment

If you reuse or adapt these materials in teaching or research,  
please provide appropriate attribution to this repository.