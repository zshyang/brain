#!/bin/bash

#SBATCH -t 1-00:00:00 #upper bound time limit for job to finish
#SBATCH -p general
#SBATCH -q public
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH -o out_err/slurm.%A_%a.out
#SBATCH -e out_err/slurm.%A_%a.err
#SBATCH --cpus-per-task=4

./simg.sh run 
