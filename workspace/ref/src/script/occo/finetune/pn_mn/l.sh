COM_PATH="/home/george/George/"
PROJECT_PATH="${COM_PATH}projects/min_modelnet/src"
DATASET_PATH="${COM_PATH}dataset/"

singularity run --nv --no-home \
-B ${PROJECT_PATH}src/:/workspace/ \
-B :/dataset/ \
-B /home/george/George/projects/min_modelnet/runtime/:/runtime/ \
/home/george/George/singularity/pnvae.sif \
/workspace/script/occo/finetune/pn_mn/s.sh
