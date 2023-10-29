#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
bop_root_path=$data_path/bop
train_root_path=$data_path/train_aae
ds_name=itodd
img_size=256
obj_id=1
eval_batch_size=10
device=mps
eval_mp_queue_size=5

train_subdir=ds_${ds_name}_obj_${obj_id}_imsz_${img_size}_20231024_214603

sdp_src_path=$code_path/sdp
segm_src_path=$code_path/lib/segmenter
export PYTHONPATH=$PYTHONPATH:$sdp_src_path:$segm_src_path

cd "$sdp_src_path" || exit 1
python c_01_run_aae.py \
  --bop-root-path $bop_root_path \
  --train-root-path $train_root_path \
  --train-subdir $train_subdir \
  --eval-batch-size $eval_batch_size \
  --device $device \
  --ds-mp-loading \
  --eval-mp-queue-size $eval_mp_queue_size


