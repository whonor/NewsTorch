#!/bin/bash
#SBATCH --account=scom.trotman
#SBATCH --job-name=TableComp
#SBATCH --partition=aoraki_gpu_H100
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --array=0-29%3
#SBATCH --output=artifacts/table_composition/slurm/%A_%a.out
set -euo pipefail
cd /projects/sciences/computing/wanro238/NewsTorch
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled WANDB_SILENT=true
/projects/sciences/computing/wanro238/envs/NewsTorch/bin/python -u \
    scripts/run_table_composition.py run --task "${SLURM_ARRAY_TASK_ID}"
