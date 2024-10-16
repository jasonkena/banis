#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00
#SBATCH --cpus-per-task=32
#SBATCH --mem=500000

mamba activate nisb

# Set TMPDIR for torch compile to avoid race conditions when runs are started in parallel
tmp_dir="/tmp/banis/${SLURM_JOB_ID}/"
mkdir -p $tmp_dir
export TMPDIR=$tmp_dir
srun python3 BANIS.py "$@"