singularity exec --nv --no-home \
--bind /scratch/zyang195/projects/brain/:/workspace/ \
/scratch/zyang195/singularity/pytorch-1-12-1.simg \
/workspace/src/data_processing/merge_dataset/scripts/run_python.sh $1
