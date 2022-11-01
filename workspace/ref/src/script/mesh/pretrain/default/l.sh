COM_PATH="/home/george/George/"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}dataset/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity exec --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/dataset/ \
-B ${RUNTIME_PATH}:/runtime/ \
/home/george/George/singularity/occo_v2.simg \
/workspace/script/mesh/pretrain/s.sh