singularity exec --nv --no-home \
--bind /scratch/zyang195/projects/brain/:/workspace/ \
--bind /scratch/zyang195/dataset/:/dataset/ \
/scratch/zyang195/singularity/pytorch-1-12-1.simg \
/workspace/src/uncertainty/sh/debug_python.sh