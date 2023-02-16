#!/bin/bash

for i in {1..3000}
do 
    sbatch batch_process.sh $i
done
