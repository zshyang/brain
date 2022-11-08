DEBUG=false
EXP_NAME="finetuning"

cd /workspace/$EXP_NAME

for VAR in 001 002 003 004 005 006 007 008 009
do
    if $DEBUG
    then
        CUDA_VISIBLE_DEVICES=0 python \
        -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
        train.py \
        --options exps/train$VAR.yaml
    else
        CUDA_VISIBLE_DEVICES=0 python \
        train.py \
        --options exps/train$VAR.yaml
    fi
done