module purge
module load singularity/3.8.0

COM_PATH="/scratch/zyang195/"
EXP_STRING="modelnet40/meshnet2/ssl_mae/0_2"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}datasets/"
SING_PATH="${COM_PATH}singularity/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity exec --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/datasets/ \
-B ${RUNTIME_PATH}:/runtime/ \
${SING_PATH}pnvae.sif \
/workspace/exps/${EXP_STRING}/s.sh

cd $SRC_PATH
python3 cont.py --outf ${EXP_STRING}

if [[ $? -eq 124 ]]; then
  sbatch ${SRC_PATH}exps/${EXP_STRING}/a.sh
fi
