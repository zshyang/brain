DEBUG=$1
INDEX=$2
PYTHON_DIR=$3

cd $PYTHON_DIR

if [ "$DEBUG" = "debug" ]; then
    python \
    -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
    ./main.py \
    --index $INDEX
else
    python \
    ./main.py \
    --index $INDEX
fi
