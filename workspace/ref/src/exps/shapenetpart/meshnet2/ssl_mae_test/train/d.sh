cd /workspace/
CUDA_VISIBLE_DEVICES=0 python \
-m debugpy --listen 0.0.0.0:5567 --wait-for-client \
-m torch.distributed.launch --nproc_per_node=1 \
train.py \
--options exps/shapenetpart/meshnet2/ssl_mae_test/train/d.yaml
