#!/bin/bash
#SBATCH --job-name=freight_rate
#SBATCH --partition=gpu_l40        # gpu_h100 also works when its GPU is free
#SBATCH --qos=limited
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python train.py --device cuda
.venv/bin/python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
