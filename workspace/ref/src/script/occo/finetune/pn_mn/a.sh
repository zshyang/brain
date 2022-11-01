#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 6            # number of "tasks" (cores)
#SBATCH --mem=10G       # GigaBytes of memory required (per node)
#SBATCH -t 0-04:00:00   # time in d-hh:mm:ss

#SBATCH -p publicgpu
#SBATCH -q wildfire       # QOS
#SBATCH -o /scratch/zyang195/projects/base/err/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /scratch/zyang195/projects/base/err/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --gres=gpu:1    # Request one GPU
#SBATCH -C V100

module purge
module load singularity/3.8.0

singularity exec --nv --no-home \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
-B /scratch/zyang195/projects/base/runtime/:/runtime/ \
/scratch/zyang195/singularity/pnvae.sif \
/workspace/script/occo/finetune/pn_mn/s.sh
