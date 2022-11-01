#!/bin/bash
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -o comp.%J.out
#SBATCH -e comp.%J.err
#SBATCH --mem=50G
#SBATCH --time=10:00:00

# Compressing with pigz:
module load pigz/2.7.0
tar cf - /scratch/zyang195/dataset/shapenet/mansim | pigz > man.tar.gz
