#!/bin/bash
#SBATCH --array=0-500
#SBATCH -t 60
#SBATCH -o oe/slurm.%j.out
#SBATCH -e oe/slurm.%j.err
module purge
module load singularity/3.8.0

singularity exec \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
/scratch/zyang195/singularity/occo_v2.simg \
/workspace/script/prep_data/mesh/s.sh \
/workspace/lib/prep_data/mesh/agave/process.py \
$SLURM_ARRAY_TASK_ID 104
