#!/bin/bash
summaries_dir=/home/mkudinov/workspace/demodulation_summaries
train_dir=/home/mkudinov/workspace/demodulation_train
data_dir=/home/mkudinov

rm -rf $summaries_dir
mkdir $summaries_dir

for test_id in {0..36} 
do 
python train.py --data_dir=$data_dir --train_dir=$train_dir --summaries_dir=$summaries_dir --test_signal_id=$test_id --valid_signal_id=0 --suffix=$test_id > log.$test_id.txt
done
