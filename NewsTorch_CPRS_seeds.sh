#!/bin/bash
#SBATCH --job-name=NewsTorch_CPRS
#SBATCH --partition=genoa
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=_CPRS_%A_%a.out
#SBATCH --array=0-4

source ~/miniforge3/bin/activate
conda activate /nesi/project/uoo04379/envs/NewsTorch

python general_runner.py --seed=${SLURM_ARRAY_TASK_ID} --DATASET_ROOT=ebnerd_demo --dataset_size=demo --dataset_name=ebnerd --model=CPRS --epoch=10 --word_embedding_dim=1024
