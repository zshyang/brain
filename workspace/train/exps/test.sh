DEBUG=true
EXP_NAME="train"

cd /workspace/$EXP_NAME

if $DEBUG 
then
    CUDA_VISIBLE_DEVICES=0 python \
    -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
    single_gpu_test.py \
    --options exps/test.yaml
else
    CUDA_VISIBLE_DEVICES=0 python \
    single_gpu_test.py \
    --options exps/test.yaml
fi