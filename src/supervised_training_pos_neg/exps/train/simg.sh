DEBUG=$1

SIMG_DIR=/workspace/src/supervised_training_pos_neg/exps/train
PYTHON_DIR=/workspace/src/supervised_training_pos_neg
OPTION_PATH=exps/train/a.yaml

if [ "$DEBUG" = "debug" ]; then
    echo "Debug mode"
else
    echo $DEBUG" mode"
fi

singularity exec --nv \
--bind /scratch/zyang195/projects/brain/:/workspace/ \
/scratch/zyang195/singularity/pytorch3d-0-4-0.simg \
$SIMG_DIR/python.sh $DEBUG $PYTHON_DIR $OPTION_PATH
