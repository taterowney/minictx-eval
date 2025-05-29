#!/bin/bash

#SBATCH --job-name=minictx_eval
#SBATCH --partition=preempt
#SBATCH --output=logs/minictx_eval.out
#SBATCH --error=logs/minictx_eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

source venv/bin/activate
python3 check_new.py --model-name "o4-mini" --dataset-name "mathlib" --num-samples 32
