DEBUG=$1
PYTHON_DIR=$2

cd $PYTHON_DIR

if [ "$DEBUG" = "debug" ]; then
    python \
    -m debugpy --listen 0.0.0.0:5566 --wait-for-client \
    ./main.py
else
    python \
    ./main.py
fi
