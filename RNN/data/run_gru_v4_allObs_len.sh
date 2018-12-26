#!/bin/bash
#shell script to rn the gru_prototype_v4.py for different observed lengths
echo "beginning to train the network... press ctrl+C to exit"

num_epochs=100
trap "exit" INT
lr=0.0007
pred_len_list=(2 4 6 8 12)
for pred_len in ${pred_len_list[@]}
do
	for obs_len in $(seq 2 1 12)
	do  
	   echo "***running learning rate of $lr with $num_epochs epochs and $obs_len obs_len***"
	   python gru_prototype_v4.py --num_epochs $num_epochs --learning_rate $lr --pred_len $pred_len --dataset_name "eth" --obs_len $obs_len
	done
done
echo "all done!"