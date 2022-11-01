cd /workspace/
python -m torch.distributed.launch --nproc_per_node=4 \
train.py --options exps/occo/pretrain/moco.yaml
