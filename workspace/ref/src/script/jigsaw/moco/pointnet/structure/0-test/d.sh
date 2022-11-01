cd /workspace/
python -m debugpy \
--listen 0.0.0.0:5566 \
--wait-for-client \
-m torch.distributed.launch \
--nproc_per_node=1 \
test.py \
--options \
exps/jigsaw/moco/pointnet/structure/0-test.yaml
