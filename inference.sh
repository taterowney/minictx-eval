#!/bin/bash

#SBATCH --job-name=minictx_eval
#SBATCH --partition=debug
#SBATCH --output=logs/minictx_eval.out
#SBATCH --error=logs/minictx_eval.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G

#vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --max_model_len 50000 --enable-reasoning --reasoning-parser deepseek_r1 > logs/vllm.out 2>&1 &
#
#echo "Waiting for vLLM server to start..."
#until curl -s http://localhost:8000/ping > /dev/null; do
#   sleep 5
#done
#
#echo "vLLM server is up and running."


source venv/bin/activate
python3 agent_boilerplate.py > logs/python.out 2>logs/python.out
