#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
bop_root_path=$data_path/bop
train_root_path=$data_path/train_aae
ds_name=itodd
img_size=256
obj_id=1
epochs=100
batch_size=10
train_subdir=last_or_new
device=cpu

sdp_src_path=$code_path/sdp
segm_src_path=$code_path/lib/segmenter
export PYTHONPATH=$PYTHONPATH:$sdp_src_path:$segm_src_path

cd $sdp_src_path
python b_01_train_aae.py \
  --bop-root-path $bop_root_path \
  --train-root-path $train_root_path \
  --train-subdir $train_subdir \
  --dataset-name $ds_name \
  --obj-id $obj_id \
  --img-size $img_size \
  --epochs $epochs \
  --batch-size $batch_size \
  --device $device

