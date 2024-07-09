#!/bin/sh

python3 lib/find_clouds.py $SNAP_DIR $OUT_DIR 0
python3 lib/build_network.py $SNAP_DIR $OUT_DIR 2900 3100 $NCORES