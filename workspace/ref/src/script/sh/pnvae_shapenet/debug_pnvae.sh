#---- debug for train ----
# while true; do
#     python -m debugpy --listen 0.0.0.0:5682 \
#     --wait-for-client \
#     -m torch.distributed.launch --nproc_per_node=1 \
#     train.py --options exps/pnvae_shapenet.yaml
# done
#---- debug the test function ----
# while true; do
#     python -m debugpy --listen 0.0.0.0:5682 \
#     --wait-for-client \
#     -m torch.distributed.launch --nproc_per_node=1 \
#     test.py --options exps/pnvae_shapenet/test.yaml
# done
#---- debug without_vae ----
while true; do
    python -m debugpy --listen 0.0.0.0:5682 \
    --wait-for-client \
    -m torch.distributed.launch --nproc_per_node=1 \
    train.py --options exps/pnvae_shapenet/without_vae.yaml
done
