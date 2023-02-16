index=$1

# python used in singularity image
cd /workspace/src/data_processing/merge_dataset

python ./merge.py --index $index
