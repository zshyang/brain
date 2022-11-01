cd workspace/
python -m torch.distributed.launch --nproc_per_node=1 \
test.py --options exps/pnvae_shapenet/test.yaml
