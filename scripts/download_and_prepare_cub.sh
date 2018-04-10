#!/bin/bash
#Dataset webpage: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

DOWNLOAD_DIR="/home/ruthfong/datasets/cub"

mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
rm CUB_200_2011.tgz

wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz
tar -xvf segmentations.tgz
rm segmentations.tgz
