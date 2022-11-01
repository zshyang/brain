#!/bin/bash

#SBATCH --array=0-500
#SBATCH -t 30                    # time in d-hh:mm:ss
#SBATCH -o out_err/slurm.%j.out     # file to save job's STDOUT (%j = JobId)
#SBATCH -e out_err/slurm.%j.err     # file to save job's STDERR (%j = JobId)

# Always purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
# source ~/.bashrc
# # module load cuda/10.2.89
# module load gcc/.6.1.0
# module load blender/2.78
# # Activate my conda environment
# conda activate 3d_detail
# # Python command line.
# cd /scratch/zyang195/decimate_mesh
module load singularity/3.8.0

singularity exec \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
-B /scratch/zyang195/data_3d_detail/:/mnt/ \
/scratch/zyang195/singularity/large-mesh.sif \
/workspace/script/prep_data/mesh/s.sh \
/workspace/lib/prep_data/agave/process_man_sim.py \
$SLURM_ARRAY_TASK_ID 106
