cd /workspace/
while true; do
    python -m debugpy --listen 0.0.0.0:5566 \
    --wait-for-client \
    -m torch.distributed.launch --nproc_per_node=4 \
    train.py --options exps/occo/pretrain/moco.yaml
done
