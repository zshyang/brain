index=$1

# python used in singularity image
cd /workspace/src/data_processing/merge_dataset/0_save_mfile_into_npz_json

python \
-m debugpy --listen 0.0.0.0:5566 --wait-for-client \
./merge.py \
--index $index
