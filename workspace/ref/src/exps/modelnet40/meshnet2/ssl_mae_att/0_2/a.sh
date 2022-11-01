#!/bin/bash
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=20G
#SBATCH -t 0-04:00:00
#SBATCH -p htcgpu1
#SBATCH -q normal
#SBATCH -o /scratch/zyang195/projects/base/err/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /scratch/zyang195/projects/base/err/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --gres=gpu:1    # Request one GPU

module purge
module load singularity/3.8.0

COM_PATH="/scratch/zyang195/"
EXP_STRING="modelnet40/meshnet2/ssl_mae_att/0_2"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}datasets/"
SING_PATH="${COM_PATH}singularity/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity exec --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/datasets/ \
-B ${RUNTIME_PATH}:/runtime/ \
${SING_PATH}pnvae.sif \
/workspace/exps/${EXP_STRING}/s.sh

cd $SRC_PATH
python3 cont.py --outf ${EXP_STRING}

if [[ $? -eq 124 ]]; then
  sbatch ${SRC_PATH}exps/${EXP_STRING}/a.sh
fi
