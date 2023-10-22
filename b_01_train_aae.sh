#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
bop_root_path=$data_path/bop
train_root_path=$data_path/train_aae
ds_name=itodd
img_size=256
obj_id=1

#epochs=3
#train_epoch_steps=20
#val_epoch_steps=10
#batch_size=5
#device=cpu
epochs=200
train_epoch_steps=-1
val_epoch_steps=-1
batch_size=10
device=mps

train_subdir=last_or_new
train_subdir=new
learning_rate=0.01

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
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --batch-size $batch_size \
  --device $device \
  --learning-rate $learning_rate


