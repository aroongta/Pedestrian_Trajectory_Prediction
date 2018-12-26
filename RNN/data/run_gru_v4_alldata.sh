#!/bin/bash
# Shell script to run gru_prototype_v4 an all data sets for pred len 2,8 and 12
echo "beginning to train the network on all datasets... press ctrl+C to exit"

## optimized hyperparameters
num_epochs=100
obs_len_list=(2 3 4 5 6 7 8 9 10 11 12)
lr=0.0007
pred_len_list=(2 3 4 5 6 7 8 9 10 11 12)
declare -a data_dirs=("eth" "zara1" "zara2" "univ" "hotel")

trap "exit" INT
for data_dir in "${data_dirs[@]}"
do
	for obs_len in ${obs_len_list[@]}
	do	
		for pred_len in ${pred_len_list[@]}
		do
			echo "***running with $num_epochs epochs $obs_len obs_len and $pred_len pred_len for $data_dir dataset***"
			python3 gru_prototype_v41.py \
			--num_epochs $num_epochs \
			--learning_rate $lr \
			--pred_len $pred_len \
			--obs_len $obs_len \
			--dataset_name $data_dir
		done
	done
done
echo "all done!"