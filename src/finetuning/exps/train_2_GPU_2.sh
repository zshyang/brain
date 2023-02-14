DEBUG=false
EXP_NAME="finetuning"

cd /workspace/$EXP_NAME

for VAR in 000 001 002 003 004
do
    if $DEBUG
    then
        CUDA_VISIBLE_DEVICES=2 python \
        -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
        train.py \
        --options exps/2/train$VAR.yaml
    else
        CUDA_VISIBLE_DEVICES=2 python \
        train.py \
        --options exps/2/train$VAR.yaml
    fi
done