cd /workspace/
python -m torch.distributed.launch --nproc_per_node=1 --master_port 47778 \
train.py --options exps/modelnet40/meshnet2/ssl_mae/0_8.yaml
