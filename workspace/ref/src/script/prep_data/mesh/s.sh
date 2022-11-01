#!/usr/bin/env bash

PROCESS=$1
INDEX=$2
INTERVAL=$3

cd /workspace/
python /workspace/lib/prep_data/agave/enterpoint.py \
--process $PROCESS --index $INDEX --interval $INTERVAL
