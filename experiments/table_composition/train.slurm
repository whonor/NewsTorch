#!/bin/bash
#SBATCH --job-name=TableComp
#SBATCH --gpus-per-node=l4:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --array=0-29%3
#SBATCH --output=artifacts/table_composition/slurm/%A_%a.out
set -euo pipefail
source ~/miniforge3/bin/activate
conda activate /nesi/project/uoo04379/envs/NewsTorch
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled WANDB_SILENT=true
python -u scripts/run_table_composition.py run --task "${SLURM_ARRAY_TASK_ID}"
