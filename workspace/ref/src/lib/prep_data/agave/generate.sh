#!/bin/bash

#SBATCH --array=0-500
#SBATCH -t 30                    # time in d-hh:mm:ss
#SBATCH -o out_err/slurm.%j.out     # file to save job's STDOUT (%j = JobId)
#SBATCH -e out_err/slurm.%j.err     # file to save job's STDERR (%j = JobId)

# Always purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
source ~/.bashrc
# module load cuda/10.2.89
module load gcc/.6.1.0
module load blender/2.78
# Activate my conda environment
conda activate 3d_detail
# Python command line.
cd /scratch/zyang195/decimate_mesh
python enterpoint.py --index $SLURM_ARRAY_TASK_ID --interval 88
