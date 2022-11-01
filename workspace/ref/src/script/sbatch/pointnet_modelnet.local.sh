singularity exec --nv --no-home \
-B /home/george/George/projects/min_modelnet/src/:/workspace/ \
-B /home/george/George/datasets/modelnet/modelnet40_ply_hdf5_2048/:/dataset/ \
-B /home/george/George/projects/min_modelnet/run/:/runtime/ \
/home/george/George/singularity/pointnet_modelnet.sif \
/workspace/sh/pointnet_modelnet.sh
