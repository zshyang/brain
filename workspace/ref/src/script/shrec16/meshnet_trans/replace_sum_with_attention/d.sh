cd /workspace/
CUDA_VISIBLE_DEVICES=0  python -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
-m torch.distributed.launch --nproc_per_node=1 \
train.py --options exps/shrec16/meshnet_trans/replace_sum_with_attention.yaml
