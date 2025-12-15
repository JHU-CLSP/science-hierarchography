#!/bin/bash

#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition nvl,h100,a100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --job-name="eval"
#SBATCH --output="eval.out"

conda activate YOUR_ENV
which python
nvidia-smi

python run.py --hierarchy /data/here/results/hierarchy.json \
--data /data/here/papers \
--output search_results.csv \
--test_count 100 \
--query_type problem \
--evaluator qwen \
--evaluate_type normal \
--run_time 5
