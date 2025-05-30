#!/bin/bash

TASK="full_proof_context"
NUM_SAMPLES=32
DATASET="mathlib"

# MODEL="l3lab/ntp-mathlib-context-deepseek-coder-1.3b"
MODEL="o4-mini"


source ./venv/bin/activate # Or however you configure your python environment
python check.py --task ${TASK} --model-name ${MODEL} --dataset-name ${DATASET} --num-samples ${NUM_SAMPLES}