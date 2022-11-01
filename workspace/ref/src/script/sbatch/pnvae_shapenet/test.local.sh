singularity run --nv --no-home \
-B /home/george/George/projects/min_modelnet/src/:/workspace/ \
-B /home/george/George/datasets/modelnet/modelnet40_ply_hdf5_2048/:/dataset/ \
-B /home/george/George/projects/min_modelnet/run/:/runtime/ \
/home/george/George/singularity/pnvae.sif \
/workspace/sh/pnvae_shapenet/test.sh
