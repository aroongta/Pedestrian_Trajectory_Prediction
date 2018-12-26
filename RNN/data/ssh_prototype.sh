#!/bin/bash
echo "beginning to train the network... press ctrl+C to exit"

num_epochs=100
trap "exit" INT
lr=0.0005

for pred_len in $(seq 2 1 12)
do  
		echo "***running learning rate of $lr with $num_epochs epochs and $pred_len pred_len***"
		python lstm_prototype_v3.py --num_epochs $num_epochs --learning_rate $lr --pred_len $pred_len
done


echo "all done!"