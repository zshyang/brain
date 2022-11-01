cd /workspace/
python -m torch.distributed.launch \
--nproc_per_node=1 \
train.py --options \
exps/occo/pretrain/jigsaw/pointnet.yaml
