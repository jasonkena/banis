#!/bin/bash -l

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000G
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --partition=p.large

mamba activate nisb

# Set TMPDIR for torch compile to avoid race conditions when runs are started in parallel
tmp_dir="/tmp/banis/${SLURM_JOBID}/"
mkdir -p $tmp_dir
export TMPDIR=$tmp_dir

resubmit_job() {
  echo "Job is being resubmitted..."
  EXP_NAME=$(echo "${LONG_JOB_ARGS}" | grep -oP '(?<=--exp_name )\S+')
  sbatch --dependency=afterany:${SLURM_JOBID} \
         --export=ALL,RESUME=TRUE,LONG_JOB=TRUE,SAVE_DIR=${SAVE_DIR},LONG_JOB_ARGS="${LONG_JOB_ARGS}" \
         --output=${SAVE_DIR}/slurm.out \
         "$0" "${@}"
  exit 0
}
trap 'resubmit_job' USR1

if ! [ -n "$LONG_JOB" ]; then
  echo "Starting a normal job"
  srun mamba run -n nisb --no-capture-output python3 -u BANIS.py "${@}"
  exit 0
fi

if [ -n "$RESUME" ]; then
    echo "Resuming from the last checkpoint"
    echo "LONG_JOB_ARGS: ${LONG_JOB_ARGS}"
    echo "ALL ARGS: ${@}"
    srun mamba run -n nisb --no-capture-output python3 -u BANIS.py --resume_from_last_checkpoint ${LONG_JOB_ARGS} &
else
    echo "Starting long training from scratch."
    export LONG_JOB_ARGS="${@}"
    echo "LONG_JOB_ARGS: ${LONG_JOB_ARGS}"
    srun mamba run -n nisb --no-capture-output python3 -u BANIS.py ${LONG_JOB_ARGS} &
fi
wait
