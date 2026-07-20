# Delhivery Shipment ETA and Delay Propagation

An interpretable machine-learning project for predicting shipment ETA, estimating delay risk, and updating downstream delay predictions from observed checkpoints. The solution uses the public Delhivery logistics dataset and follows a chronological, leakage-safe evaluation protocol.

**Author:** Nguyen Nhat Long  
**Kaggle notebook:** https://www.kaggle.com/code/nhatlong1103/delhivery-prediction

## Project Objectives

The project addresses three related tasks:

1. **ETA prediction** - predict the full trip duration and final arrival time.
2. **Severe-delay classification** - estimate whether proxy delay exceeds 120 minutes.
3. **Delay propagation** - predict the remaining delay as new checkpoints are observed.

Delay is measured relative to the OSRM routing-time baseline. It is a modeling proxy, not an official Delhivery service-level agreement.

## Repository Structure

```text
.
├── README.md
├── safiri_take_home_problem_4.pdf
├── delhivery-prediction.ipynb
├── Nguyen_Nhat_Long_ETA_Shipment_Report.pdf
└── Experiments/
    ├── Top10_Model_Experiements.pdf
    ├── delhivery-baseline-tree-models.ipynb
    ├── delhivery-boosting-and-linear-models-benchmark.ipynb
    ├── delhivery-extra-trees-rotation-forest-bagging-ensemble.ipynb
    ├── delhivery-full-regressor-propagation-benchmark.ipynb
    ├── delhivery-time-series-models.ipynb
    ├── delhivery-sequential-checkpoint-models.ipynb
    ├── delhivery-graph-neural-network-models.ipynb
    └── delhivery-final-extra-trees-model.ipynb
```

### Main Submission Files

- **`safiri_take_home_problem_4.pdf`** - original Safiri AI take-home assignment.
- **`delhivery-prediction.ipynb`** - final executable solution and source of the submitted results.
- **`Nguyen_Nhat_Long_ETA_Shipment_Report.pdf`** - final three-page technical report.

### Experiment Files

The `Experiments/` directory records the model-selection process:

| File | Purpose |
|---|---|
| `delhivery-baseline-tree-models.ipynb` | Benchmarks Decision Tree, Random Forest, Extra Trees, AdaBoost, and XGBoost. |
| `delhivery-boosting-and-linear-models-benchmark.ipynb` | Extends the benchmark with boosting, linear, robust, SVM, probabilistic, and ensemble models. |
| `delhivery-extra-trees-rotation-forest-bagging-ensemble.ipynb` | Compares Extra Trees, Rotation Forest, Bagging, and a stacking/boosting ensemble. |
| `delhivery-full-regressor-propagation-benchmark.ipynb` | Applies the full regressor registry to checkpoint-level delay propagation. |
| `delhivery-time-series-models.ipynb` | Tests classical and deep time-series approaches such as SARIMAX, Prophet, DeepAR, N-BEATS, TFT, PatchTST, and TimesNet. |
| `delhivery-sequential-checkpoint-models.ipynb` | Tests RNN, LSTM, GRU, TCN, Transformer, Informer, and Autoformer sequence models. |
| `delhivery-graph-neural-network-models.ipynb` | Models observed checkpoint prefixes as graphs using eight GNN architectures. |
| `delhivery-final-extra-trees-model.ipynb` | Focused Extra Trees implementation used to consolidate the final pipeline. |
| `Top10_Model_Experiements.pdf` | Summarizes the ten strongest completed models for each prediction task. |

## Solution Architecture

```text
Delhivery checkpoint records
          │
          ▼
Data validation and preprocessing
- timestamp parsing
- missing location recovery
- anomaly and quality checks
          │
          ▼
Hierarchical aggregation
checkpoint rows → route legs → one row per trip
          │
          ├─────────────── Start-of-trip branch ───────────────┐
          │                                                    │
          │  planned route, time, distance, and location       │
          │           │                                        │
          │           ├── Extra Trees Regressor → ETA/duration │
          │           └── Extra Trees Classifier → delay risk  │
          │                                                    │
          └──────────── Checkpoint-observation branch ─────────┤
                      observed cumulative progress             │
                               │                               │
                               └── Extra Trees Regressor       │
                                   → remaining/final delay     │
                                                               ▼
                         Evaluation, SHAP interpretation,
                         example prediction, and artifact export
```

### Processing and Evaluation Flow

1. The raw data is cleaned and aggregated from checkpoint rows to legs and then trips.
2. Start-of-trip features use only information available before departure.
3. The external training period is split chronologically into fit and validation sets; the external test period remains isolated.
4. Categorical features are one-hot encoded, numeric features are imputed and scaled, and transformed matrices remain sparse.
5. Candidate models are selected only on validation performance.
6. Final models are evaluated once on the held-out test set.
7. Global feature importance and local SHAP explanations identify the main contributors to ETA and delay predictions.

## Final Models and Results

| Task | Final model | Main final-test result |
|---|---|---|
| ETA prediction | Extra Trees Regressor | MAE **104.93 min**, R² **0.898** |
| Severe-delay classification | Extra Trees Classifier | F1 **0.899**, ROC-AUC **0.911** |
| Delay propagation | Extra Trees Regressor | MAE improves from **100.80 min** near early progress to **85.95 min** near completion |

The experiment benchmark also identified Extra Trees as the strongest completed validation model for all three tasks under the documented protocols.

## Running the Final Notebook

1. Open `delhivery-prediction.ipynb` in Kaggle or Jupyter.
2. Make `delhivery_data.csv` available at the configured Kaggle path, in the project root, or inside a local `data/` directory.
3. Install the core dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib shap
```

4. Run all cells from top to bottom.

After execution, the notebook creates an `artifacts/` directory containing fitted preprocessors and models, evaluation tables, feature-importance files, predictions, and metadata.

## Methodological Safeguards

- chronological splitting with no temporal shuffling;
- no overlap of `trip_uuid` across training and final test data;
- removal of future-only variables from start-of-trip prediction;
- validation-only model and probability-threshold selection;
- checkpoint propagation based only on information already observed;
- one nearest checkpoint per trip for progress-based evaluation;
- consistent delay calculation from the ETA prediction.
