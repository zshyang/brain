#!/bin/bash

#SBATCH --job-name=array
# SBATCH --array=1-300 #number of total jobs to spawn
#SBATCH -t 0-00:10:00 #upper bound time limit for job to finish
#SBATCH -p general
#SBATCH -q public
# SBATCH --ntasks=10 #number of concurrently running jobs
#SBATCH -o out_err/slurm.%A_%a.out
#SBATCH -e out_err/slurm.%A_%a.err
#SBATCH --mail-user=zyang195@asu.edu
#SBATCH --mail-type=ALL

./run_simg.sh $1
