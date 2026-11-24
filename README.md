# Explainable Knowledge-Tracing-Based Adaptive Learning for Personalized Pedagogical Recommendation

This repository contains the code, configurations, frozen data split and result files for the paper
*An Explainable Knowledge-Tracing-Based Adaptive Learning Framework for Personalized Pedagogical Recommendation*.

The framework predicts whether a learner will answer the next question correctly. It then ranks five pedagogical strategies: remedial, example-based, practice-heavy, step-by-step and challenge-based. Each recommendation comes with a SHAP explanation.

## Repository structure

| Folder | Content |
|---|---|
| `src/adaptive_kt/` | Python package: data processing, causal features, models, training, evaluation, SHAP tools |
| `src/adaptive_kt/models/behavior_aware_kt_v3.py` | Proposed model (attention backbone) |
| `src/adaptive_kt/models/sequence_v3.py` | Causal learner state and online difficulty-adjusted (Elo-style) mastery |
| `src/adaptive_kt/models/hgakt_kt3.py` | HGAKT+PR base model (prediction component, our adaptation) |
| `src/adaptive_kt/models/` (others) | Baselines: DKT, SAKT, AKT, SAINT+, Transformer-KT, logistic regression, random forest, XGBoost |
| `configs/` | Experiment settings (EdNet-KT3, ASSISTments 2017, smoke test) |
| `scripts/` | Data preparation, training, baselines, pedagogy recommender, tables and figures |
| `kaggle/` | The notebook cells that ran the final experiments (EdNet-KT3, ASSISTments 2017, pedagogy) |
| `results/kt3_final/`, `results/assist2017/` | Final tables: main results, significance tests, ablation, model selection, training checks |
| `results/ednet_kt3_trackB/split_assignments.csv` | Frozen learner-disjoint 70/10/20 split used by every model |
| `figures/` | Result figures and architecture diagrams used in the paper |

In the code, `v3` is the internal name of the final proposed model. `v1` and `v2` are earlier versions, and the final model imports them.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or requirements-lock.txt for the exact versions we used
export PYTHONPATH=$PWD/src
bash scripts/smoke_test.sh               # runs in about a minute on synthetic data
```

`ENVIRONMENT.txt` lists the Python, PyTorch, CUDA and GPU versions of the original runs.

## Data

The datasets are not redistributed in this repository. Download them from the official sources:

* **EdNet-KT3:** `bash scripts/download_ednet_primary.sh`, which downloads the official EdNet files.
* **ASSISTments 2017:** downloaded automatically by `kaggle/assist17_cell.py`. You can also download it from the official ASSISTments data page and place it in `data/raw/assistments2017/`.

## Reproducing the results

The model and design choices were made on validation learners only. The test learners were evaluated once.

| Step | EdNet-KT3 | ASSISTments 2017 |
|---|---|---|
| 1. Prepare data and features | `python scripts/prepare_ednet_kt3.py --config configs/ednet_kt3_trackB.yaml` | `python scripts/kt_a17.py data` |
| 2. Validation-only design selection, 3 seeds, ablation, baselines, bootstrap | `python kaggle/kt3_cell.py` | `python kaggle/assist17_cell.py` |
| 3. Pedagogical recommender and SHAP | `python scripts/kt_pedagogy.py final` | `KT_PED_PREFIX=assist2017 python scripts/kt_pedagogy.py final` |
| 4. Tables and figures | `python scripts/make_tables.py results/kt3_final results/assist2017 paper_tables` and `python scripts/kt_figures_clean.py` | same |

Seeds: 42, 52 and 62. The split file in `results/ednet_kt3_trackB/` fixes which learner belongs to training, validation and test.

## Main results (test AUC)

| Model | EdNet-KT3 | ASSISTments 2017 |
|---|---|---|
| Proposed | 0.7742 ± 0.0003 | 0.8481 ± 0.0002 |
| HGAKT (base model, our adaptation) | 0.7587 | 0.8131 |
| XGBoost with the same inputs | 0.7615 | 0.7705 |
| AKT | 0.7566 | 0.7747 |
| DKT | 0.7489 | 0.7943 |

All AUC gains of the proposed model are significant at p ≤ 0.001 (paired learner-level bootstrap, 1,000 resamples). The complete tables are in `results/`.

## Note on exact reproducibility

Seeds are fixed and deterministic cuDNN is enabled (`src/adaptive_kt/utils/reproducibility.py`). Some GPU operations are still not bit-exact across different GPUs, drivers or library versions. A rerun on different hardware can therefore differ in the third or fourth decimal place. For example, the proposed model's test AUC varied by only ±0.0003 across seeds, while its smallest gap to a baseline was 0.0127. The XGBoost, random forest and logistic regression results are deterministic for a fixed seed on CPU.

