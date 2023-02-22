DEBUG=$1

SIMG_DIR=/workspace/src/data_processing/merge_dataset/2_generate_meshnet_feature/
SIMG_SCRIPTS_DIR=$SIMG_DIR"scripts/"

if [ "$DEBUG" = "debug" ]; then
    echo "Debug mode"
else
    echo $DEBUG" mode"
fi

singularity exec --nv \
--bind /scratch/zyang195/projects/brain/:/workspace/ \
/scratch/zyang195/singularity/pytorch3d-0-4-0.simg \
$SIMG_SCRIPTS_DIR/python.sh $DEBUG $SIMG_DIR
