COM_PATH="/home/george/George/"

PROJECT_PATH="${COM_PATH}projects/base/"
DATASET_PATH="${COM_PATH}dataset/"

SRC_PATH="${PROJECT_PATH}src/"
RUNTIME_PATH="${PROJECT_PATH}runtime/"

singularity exec --nv --no-home \
-B ${SRC_PATH}:/workspace/ \
-B ${DATASET_PATH}:/dataset/ \
-B ${RUNTIME_PATH}:/runtime/ \
/home/george/George/singularity/occo.simg \
/workspace/script/occo/pretrain/p/s.sh

cd $SRC_PATH
python3 cont.py --outf occo/pretrain/p

if [[ $? -eq 124 ]]; then
    /home/george/George/projects/base/src/script/occo/pretrain/p/l.sh
fi
