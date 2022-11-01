cd /workspace/
CUDA_VISIBLE_DEVICES=1 python \
-m torch.distributed.launch --nproc_per_node=1 \
train.py --options exps/shrec16/meshnet_trans/replace_sum_with_attention.yaml
