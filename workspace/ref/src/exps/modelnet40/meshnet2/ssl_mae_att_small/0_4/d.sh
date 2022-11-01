cd /workspace/
CUDA_VISIBLE_DEVICES=0 python \
-m debugpy --listen 0.0.0.0:5567 --wait-for-client \
-m torch.distributed.launch --nproc_per_node=1 \
train.py \
--options exps/modelnet40/meshnet2/ssl_mae_att_small/0_4.yaml