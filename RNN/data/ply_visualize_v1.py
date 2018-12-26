# Script to fit polynomials to each trajecotry of a dataset and visualize the variation of coefficients
#written by: Ashish Roongta, December 2018
#Carnegie Mellon University

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import trajectories
import loader
import argparse

# building argparser
parser=argparse.ArgumentParser()
#DataSet Arguments
parser.add_argument('--dataset_name',default='eth',type=str)
parser.add_argument('--loader_num_workers',default=4,type=int)
parser.add_argument('--batch_size',default=5, type=int)
parser.add_argument('--delim',default='\t')
parser.add_argument('--obs_len',default=8,type=int)
parser.add_argument('--pred_len',default=8,type=int)
parser.add_argument('--skip',default=1,type=int)
args=parser.parse_args()

cur_dataset=args.dataset_name
data_dir = os.path.join('/mnt/d/ped_trajectory_prediction/sgan_ab/scripts/datasets/', cur_dataset + '/train')
test_data_dir=os.path.join('/mnt/d/ped_trajectory_prediction/sgan_ab/scripts/datasets/',cur_dataset+'/test')
def polyfit_visualize(args,data_path):
	degree=2 	#degree of polynomial to be fitted
	coeff=np.ones(3)
	obs_len=args.obs_len
	pred_len=args.pred_len
	dataset,dataloader=loader.data_loader(args,data_path)
	for i,batch in enumerate(dataloader):
		observed_batch=batch[0].numpy()		#observed trajectory batch
		target_batch=batch[1].numpy() 	#Target trajectory batch
		trajec=observed_batch+target_batch
		seq,peds,coords=observed_batch.shape
		for j in range(peds):
			z=np.polyfit(observed_batch[:,j,0],observed_batch[:,j,1],degree)
			coeff=np.column_stack((coeff,z))
		if(i>15):
			break
	plt.figure()
	for i in range(degree+1):
		plt.plot(coeff[i,1:],label='{} order coefficient'.format(i))
	plt.xlabel('Pedestrian Trajectories')
	plt.ylabel('Coefficients of Plynomial')
	plt.title("Fitted polynomial to dataset:{}".format(cur_dataset))	
	plt.legend()
	# plt.show(block=True)
	plt.savefig("./saved_figs/" + 'lstm_polyfit_'+cur_dataset+'predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
def main(args):
	polyfit_visualize(args,test_data_dir)

if __name__== '__main__':
	main(args)
