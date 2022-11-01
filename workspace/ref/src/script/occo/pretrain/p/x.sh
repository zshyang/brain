#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=v100-16:2

COM_PATH="/ocean/projects/mcb170053p/zyang5/"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}dataset/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity exec --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/dataset/ \
-B ${RUNTIME_PATH}:/runtime/ \
/scratch/zyang195/singularity/occo.simg \
/workspace/script/occo/pretrain/p/s.sh
