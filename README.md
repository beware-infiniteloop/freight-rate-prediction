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

