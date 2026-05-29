#!/bin/bash
#SBATCH --job-name=TCCM
#SBATCH --partition=aoraki_gpu_L4_24GB
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=_TCCM_demo_%A_%a.out
#SBATCH --array=0

source ~/miniforge3/bin/activate
conda activate NewsTorch

python general_runner.py --seed=${SLURM_ARRAY_TASK_ID} --DATASET_ROOT=ebnerd_demo --dataset_size=demo --dataset_name=ebnerd --model=TCCM --epoch=10 --word_embedding_dim=1024
