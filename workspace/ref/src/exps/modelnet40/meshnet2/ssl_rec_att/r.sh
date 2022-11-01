cd /workspace/
CUDA_VISIBLE_DEVICES=0 python \
-m torch.distributed.launch --nproc_per_node=1 \
train.py \
--options exps/modelnet40/meshnet2/ssl_rec_att.yaml
