#!/bin/bash
echo "beginning to train the network... press ctrl+C to exit"

num_epochs=100
trap "exit" INT
learn_rates=(0.0017)
pred_len_list=(2 8)
batch_size=5
for lr in ${learn_rates[@]}
do
	for pred_len in ${pred_len_list[@]}
	do  
   		echo "***running learning rate of $lr with $num_epochs epochs and $pred_len pred_len***"
   		python lstm_prototype_v3.py --num_epochs $num_epochs --learning_rate $lr --pred_len $pred_len --batch_size $batch_size
	done
done


echo "all done!"