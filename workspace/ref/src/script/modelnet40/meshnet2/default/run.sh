cd /workspace/
CUDA_VISIBLE_DEVICES=0,1 python \
-m torch.distributed.launch --nproc_per_node=2 \
train.py --options exps/modelnet40/meshnet2/default.yaml
