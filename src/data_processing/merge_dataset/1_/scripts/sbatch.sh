#!/bin/bash

#SBATCH -t 0-10:00:00 #upper bound time limit for job to finish
#SBATCH -p general
#SBATCH -q public
#SBATCH -o out_err/slurm.%A_%a.out
#SBATCH -e out_err/slurm.%A_%a.err

./simg.sh run $1
