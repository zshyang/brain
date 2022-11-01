cd /workspace/
CUDA_VISIBLE_DEVICES=0 python \
-m torch.distributed.launch --nproc_per_node=1 \
test.py \
--options exps/shapenetpart/meshnet2/ssl_mae_test/d.yaml
