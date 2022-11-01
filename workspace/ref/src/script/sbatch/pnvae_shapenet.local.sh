singularity run --nv --no-home \
-B /home/george/George/projects/min_modelnet/src/:/workspace/ \
-B /home/george/George/datasets/shapenet/shape_net_core_uniform_samples_2048/:/dataset/ \
-B /home/george/George/projects/min_modelnet/run/:/runtime/ \
/home/george/George/singularity/pnvae.sif \
/workspace/sh/train_pnvae.sh
