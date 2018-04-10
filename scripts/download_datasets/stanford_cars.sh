#!/bin/bash
#Dataset website: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

DOWNLOAD_DIR=/data/datasets/cars

# make directory (and intermediary directories) if it doesn't exist
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

# download training images
wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
tar -xvf cars_train.tgz
rm cars_train.tgz

# downlaod test images
wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
tar -xvf cars_test.tgz
rm car_test.tgz

# download dev kit
wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
tar -xvf car_devkit.tgz
rm car_devkit.tgz

# download BMW10 dataset
wget http://imagenet.stanford.edu/internal/car196/bmw10_release.tgz
tar -xvf bmw10_release.tgz
rm bmw10_release.tgz
