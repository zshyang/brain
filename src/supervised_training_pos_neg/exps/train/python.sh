DEBUG=$1
PYTHON_DIR=$2
OPTION_PATH=$3

cd $PYTHON_DIR

if [ "$DEBUG" = "debug" ]; then
    python \
    -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
    ./train.py --options $OPTION_PATH
else
    python \
    ./train.py --options $OPTION_PATH
fi
