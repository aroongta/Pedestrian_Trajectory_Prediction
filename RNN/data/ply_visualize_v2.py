# Script to fit polynomials to each trajecotry and visualize the variation of each coefficient for all datasets
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
degree=4 	#degree of polynomial to be fitted
obs_len=args.obs_len #storing the observed trajectory length from the argparser
pred_len=args.pred_len #storing the predicted trajectory length from argparser

def polyfit_visualize(args,data_path):
	coeff=np.ones(3)
	ped_c=0
	dataset,dataloader=loader.data_loader(args,data_path)
	for i,batch in enumerate(dataloader):
		observed_batch=batch[0].numpy()		#observed trajectory batch
		target_batch=batch[1].numpy() 	#Target trajectory batch
		trajec=observed_batch+target_batch
		seq,peds,coords=observed_batch.shape
		ped_c+=peds
		for j in range(peds):
			z=np.polyfit(observed_batch[:,j,0],observed_batch[:,j,1],degree)
			coeff=np.column_stack((coeff,z)) #adding the coefficients of each polynomial fitted to columns of coeff array
		if(ped_c>60):  #Just a random cap on the number of pedestrians in each dataset
			break
	# plt.figure()
	# for i in range(degree+1):
	# 	plt.plot(coeff[i,1:],label='{} order coefficient'.format(i))
	# plt.xlabel('Pedestrian Trajectories')
	# plt.ylabel('Coefficients of Plynomial')
	# plt.title("Fitted polynomial to dataset:{}".format(cur_dataset))	
	# plt.legend()
	# # plt.show(block=True)
	# plt.savefig("./saved_figs/" + 'lstm_polyfit_'+cur_dataset+'predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
	return coeff

def main(args):
	datasets=['eth','zara1','zara2','univ','hotel']
	coefficients={} #initializing an empty dictionary
	for dataset in datasets:
		print("Running on dataset: {}".format(dataset))
		test_data_dir=os.path.join('/mnt/d/ped_trajectory_prediction/sgan_ab/scripts/datasets/',dataset+'/test')
		coefficients[dataset]=polyfit_visualize(args,test_data_dir) #appending coefficients array of each dataset to the dictionary with keys=dataset_name
	for i in range(degree+1):
		plt.figure(i)
		for dataset in datasets:
			plt.plot(coefficients[dataset][i,1:],label='Dataset:{}'.format(dataset))
		plt.xlabel("Pedestrians")
		plt.ylabel("Coefficent values")
		plt.title("order of coefficient: {}".format(i))
		plt.legend()
		plt.savefig("./saved_figs/"+"poly_fit_allDatasets"+"_coeffOrder_"+str(i)+'predlen_' + str(pred_len) +'_obs'+str(obs_len)+'.png')
		print("Saved Plot for coefficient: {}".format(i))
if __name__== '__main__':
	main(args)
