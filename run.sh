#!/bin/sh

SNAP_DIR=/path/to/simulation/output/
OUT_DIR=${SNAP_DIR}allclouds/

python3 lib/find_clouds.py $SNAP_DIR $OUT_DIR 0
python3 lib/build_network_ids.py $SNAP_DIR $OUT_DIR 200 300