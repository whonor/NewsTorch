#!/bin/bash
#SBATCH --account=scom.trotman
#SBATCH --job-name=TableReport
#SBATCH --partition=aoraki
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/table_composition/slurm/report_%j.out
set -euo pipefail
cd /projects/sciences/computing/wanro238/NewsTorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/projects/sciences/computing/wanro238/envs/NewsTorch/bin/python \
    scripts/run_table_composition.py report --allow-incomplete
