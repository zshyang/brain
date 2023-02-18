#!/bin/bash

for i in {0..5}
do 
    sbatch sbatch.sh $i
done
