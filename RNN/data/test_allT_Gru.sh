#!/bin/bash
# Script to run gru_loadNtest.py for all prediction lens
echo "beginning to train the network... press ctrl+C to exit"

num_epochs=100
trap "exit" INT
lr=0.0025
data_dir="zara1"

for pred_len in $(seq 2 1 12)
do  
		echo "***running learning rate of $lr with $num_epochs epochs and $pred_len pred_len***"
		python gru_loadNtest.py --num_epochs $num_epochs --learning_rate $lr --pred_len $pred_len --dataset_name $data_dir
done


echo "all done!"