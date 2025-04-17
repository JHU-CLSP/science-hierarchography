#!/bin/bash

#SBATCH -A dkhasha1
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --partition nvl
#SBATCH --exclude=c001,c003,c005,h12,n04,n06,n05,n16,n10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name="batch"
#SBATCH --output="embedding.out"
#SBATCH --mail-type BEGIN 
#SBATCH --mail-user 1542060768@qq.com

source /home/mgao38/.bashrc
conda activate needle
which python
nvidia-smi

# python generate.py --input_folder /weka/scratch/dkhasha1/mgao38/taxonomy/dataset_10k --output_file ./embeddings/qwen_key_10k_embeddings.pkl

python main.py \
--embedding_generator qwen \
--summary_generator llama \
--clustering_method kmeans \
--evaluator llama \
--clustering_direction top_down \
--base_path /weka/scratch/dkhasha1/mgao38/taxonomy \
--cluster_sizes 1250 276 40 6 \
--run_time 1 \
--evaluate_time 1 \
--test_count 5 \
--pre_generated_embeddings_file ./embeddings/qwen_key_10k_embeddings.pkl \
--evaluate_type normal \
--embedding_source problem


