#!/usr/bin/env bash

SOURCE_DIR="/scratch/shared/nfs1/ruthfong/imagenet12"
NEW_DIR="$TMPDIR/imagenet12"

rsync -avz $SOURCE_DIR $NEW_DIR
