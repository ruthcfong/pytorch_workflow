#!/usr/bin/env bash

SOURCE_DIR="/scratch/shared/nfs1/ruthfong/ILSVRC2012/images"
NEW_DIR="$TMPDIR/imagenet12"

mkdir $NEW_DIR
rsync -avz "$SOURCE_DIR/train" "$NEW_DIR/train"
rsync -avz "$SOURCE_DIR/val_pytorch" "$NEW_DIR/val"
