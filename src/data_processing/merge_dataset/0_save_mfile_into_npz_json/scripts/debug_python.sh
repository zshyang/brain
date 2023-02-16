index=$1

# python used in singularity image
cd /workspace/src/data_processing/merge_dataset

python \
-m debugpy --listen 0.0.0.0:5566 --wait-for-client \
./merge.py \
--index $index
