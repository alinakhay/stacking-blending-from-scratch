# Stacking and Blending from Scratch

A compact experiment that implements a scikit-learn-style stacking estimator, visualises decision surfaces and compares ensemble strategies on a noisy nonlinear classification problem.

The useful result is not simply that an ensemble “wins.” On the saved 10,000-observation two-moons experiment, stacking delivered only a marginal improvement over the strongest individual learner. That is a more realistic demonstration of ensemble modelling: additional complexity has to earn its place.

## Experiment

- **Dataset:** synthetic two-moons classification with noise, split 70/30
- **Base learners:** k-nearest neighbours, ridge, random forest and LightGBM variants
- **Meta-learner:** ridge regression
- **Strategies:** a held-out blending split and out-of-fold predictions produced with `cross_val_predict`
- **Evaluation:** holdout ROC AUC and plotted decision surfaces

## Saved results

| Model | Holdout ROC AUC |
| --- | ---: |
| Random forest, depth 5 | 0.89974 |
| LightGBM, depth 2 | 0.90018 |
| Held-out stacking | 0.89892 |
| Out-of-fold stacking | **0.90048** |

The out-of-fold stack improved on the strongest displayed base learner by roughly 0.0003 AUC—too small to claim a meaningful practical advantage without repeated validation.

## What the notebook demonstrates

- Building meta-features from multiple heterogeneous estimators
- Keeping meta-model training separate from base-model fitting
- Using out-of-fold predictions to reduce leakage
- Comparing decision boundaries, not just aggregate scores
- Testing sensitivity to the number of folds and repeated blending splits

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab stacking_example.ipynb
```

LightGBM requires an OpenMP runtime; on macOS, install `libomp` with Homebrew if it is not already present.

## Limitations

This is an educational legacy notebook rather than a production benchmark. It uses a synthetic two-dimensional dataset, a single primary train/test split and regression estimators as continuous scoring models. The notebook interface and explanatory text are partly in Russian, while this README provides the English portfolio summary. A production comparison should add repeated stratified cross-validation, calibrated classifiers, uncertainty estimates and explicit runtime trade-offs.
