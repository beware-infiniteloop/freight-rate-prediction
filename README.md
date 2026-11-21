# Freight Rate Forecasting

Predicts `posted_rate` for 12,000 truck loads in **Nov–Dec 2025** from 48,000 labelled loads dated **Jan–Oct 2025**.

| | |
|---|---|
| **Model** | `log(rate) = linear model (market index, time trend, quarter-end ramp) + XGBoost (lane / load features)` |
| **Validation** | Walk-forward: learn from the past, predict the next 2 months, repeated at 3 cut-offs |
| **Result** | **1.62% MAPE** (≈ $39 per load) vs 2.90% for plain XGBoost; noise floor 1.25% |

## Submission files

| File | |
|---|---|
| [`report.docx`](report.docx) | Report: data validation, train/test split, model selection, results, December chart |
| [`validation_predictions.csv`](validation_predictions.csv) | `load_id,predicted_rate` for all 12,000 loads |
| [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) | Fixed December chart from `score.py` |
| [`data/december_chart_inputs.csv`](data/december_chart_inputs.csv) | The 31 December rows with `predicted_rate` filled |
| [`freight_rate_walkthrough.ipynb`](freight_rate_walkthrough.ipynb) | **Start here to review the work:** every analysis and experiment, step by step, with outputs and conclusions |

## Code

| File | What it does |
|---|---|
| `train.py` | Cleaning, features, walk-forward backtest, final model, writes both prediction files |
| `eda.py` | Prints the key data findings, saves plots to `outputs/` |
| `score.py` | Provided scorer: validates the output files and draws the December chart |
| `make_report.py` | Builds `report.docx` (figures in `outputs/report_figures/`) |
| `run_gpu.sbatch` | Slurm job: `train.py` on a GPU, then `score.py` |

## Setup

The assessment data is not committed. Place the provided files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv      # train.py fills its predicted_rate column
```

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # without a CUDA-12 GPU, replace `xgboost-cu12` with `xgboost`
```

## Run

```bash
# Slurm GPU cluster
sbatch run_gpu.sbatch                        # log goes to logs/

# Or directly (≈20 s on a GPU, a few minutes on CPU)
.venv/bin/python train.py --device cuda      # or --device cpu
.venv/bin/python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

# Optional: full walkthrough notebook, then the report
.venv/bin/jupyter nbconvert --to notebook --execute --inplace freight_rate_walkthrough.ipynb
.venv/bin/python eda.py
.venv/bin/python make_report.py
```
