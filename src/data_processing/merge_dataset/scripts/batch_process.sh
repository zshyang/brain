#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-20 #number of total jobs to spawn
#SBATCH --time=0-00:10:00 #upper bound time limit for job to finish
#SBATCH --p general
#SBATCH --q public
#SBATCH --ntasks=10 #number of concurrently running jobs

srun -n 1 ./the_work.sh $SLURM_ARRAY_TASK_ID
