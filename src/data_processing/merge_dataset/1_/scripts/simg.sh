DEBUG=$1
INDEX=$2

SIMG_DIR=/workspace/src/data_processing/merge_dataset/1_/
SIMG_SCRIPTS_DIR=$SIMG_DIR"scripts/"

if [ "$DEBUG" = "debug" ]; then
    echo "Debug mode"
else
    echo $DEBUG" mode"
fi

singularity exec --no-home \
--bind /scratch/zyang195/projects/brain/:/workspace/ \
/scratch/zyang195/singularity/pytorch-1-12-1.simg \
$SIMG_SCRIPTS_DIR/python.sh $DEBUG $INDEX $SIMG_DIR

# to fix the one not generated from ./sol.sh
# ./simg.sh debug 106
# 106 ==> 849
# 155 ==> 830
# 707 ==> 1259
# 798 ==> 1272
# 1017 ==> 560
# 1062 ==> 524
# 1584 ==> 1188
