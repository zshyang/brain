#!/bin/bash

for i in {0..2295}
do 
    sbatch sbatch.sh $i
done
