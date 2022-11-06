DEBUG=false
EXP_NAME="train"

cd /workspace/$EXP_NAME

if $DEBUG 
then
    CUDA_VISIBLE_DEVICES=0 python \
    -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
    train.py \
    --options exps/train000.yaml
else
    CUDA_VISIBLE_DEVICES=0 python \
    train.py \
    --options exps/train000.yaml
fi