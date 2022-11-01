COM_PATH="/home/george/George/"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}datasets/"
SING_PATH="${COM_PATH}singularity/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity run --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/datasets/ \
-B ${RUNTIME_PATH}:/runtime/ \
${SING_PATH}pnvae.sif \
/workspace/exps/modelnet40/meshnet2/ssl_mae/0_2/s.sh
