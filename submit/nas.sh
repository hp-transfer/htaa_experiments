#!/bin/bash

#SBATCH --time 0-02:00
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpus:0
#SBATCH --mem-per-cpu=8000

#SBATCH --job-name NAS

source ~/.conda/bin/activate htaa

ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "${ARGS_FILE}")
python -m hp_transfer_aa_experiments.run ${ARGS}
