#!/bin/bash
#LSUN_PYTHON_DIR=/home/ruthfong/lsun
#DOWNLOAD_DIR=/data/datasets/lsun
DOWNLOAD_DIR=/scratch/shared/slow/ruthfong/datasets/lsun
#mkdir -p $DOWNLOAD_DIR
#cd $LSUN_PYTHON_DIR
#python download.py -o $DOWNLOAD_DIR
cd $DOWNLOAD_DIR
unzip bedroom_train_lmdb.zip
unzip bedroom_val_lmdb.zip
unzip bridge_train_lmdb.zip
unzip bridge_val_lmdb.zip
unzip church_outdoor_train_lmdb.zip
unzip church_outdoor_val_lmdb.zip
unzip classroom_train_lmdb.zip
unzip classroom_val_lmdb.zip
unzip conference_room_train_lmdb.zip
unzip conference_room_val_lmdb.zip
unzip dining_room_train_lmdb.zip
unzip dining_room_val_lmdb.zip
unzip kitchen_train_lmdb.zip
unzip kitchen_val_lmdb.zip
unzip living_room_train_lmdb.zip
unzip living_room_val_lmdb.zip
unzip restaurant_train_lmdb.zip
unzip restaurant_val_lmdb.zip
unzip test_lmdb.zip
unzip tower_train_lmdb.zip
unzip tower_val_lmdb.zip
