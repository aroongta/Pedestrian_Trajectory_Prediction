#Vanialla LSTM architecture with a weight updation after each step of prediction 
# rather than after predtiction of entire sequence
#including relu activaton and dropout after the first linear layer
# prototype of vanilla_lstm network for pedestrian modeling
# written by: Bryan Zhao and Ashish Roongta, Fall 2018
# carnegie mellon university

# import relevant libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import numpy as np
import trajectories
import loader
import argparse
import gc
import logging
import os
import sys
import time
import matplotlib.pyplot as plt 

# build argparser
parser = argparse.ArgumentParser()

parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=2)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                 help='size of RNN hidden state')
# size of each batch parameter
parser.add_argument('--batch_size', type=int, default=5,
                 help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=20,
                 help='RNN sequence length')
parser.add_argument('--pred_length', type=int, default=12,
                 help='prediction length')
# number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=20,
                 help='number of epochs')
# frequency at which the model should be saved parameter
parser.add_argument('--save_every', type=int, default=400,
                 help='save frequency')
# gradient value at which it should be clipped
parser.add_argument('--grad_clip', type=float, default=10.,
                 help='clip gradients at this value')
# learning rate parameter
parser.add_argument('--learning_rate', type=float, default=0.003,
                 help='learning rate')
# decay rate for the learning rate parameter
parser.add_argument('--decay_rate', type=float, default=0.95,
                 help='decay rate for rmsprop')
# dropout probability parameter
parser.add_argument('--dropout', type=float, default=0.5,
                 help='dropout probability')
# dimension of the embeddings parameter
parser.add_argument('--embedding_size', type=int, default=64,
                 help='Embedding dimension for the spatial coordinates')
# size of neighborhood to be considered parameter
parser.add_argument('--neighborhood_size', type=int, default=32,
                 help='Neighborhood size to be considered for social grid')
# size of the social grid parameter
parser.add_argument('--grid_size', type=int, default=4,
                 help='Grid size of the social grid')
# maximum number of pedestrians to be considered
parser.add_argument('--maxNumPeds', type=int, default=27,
                 help='Maximum Number of Pedestrians')

# lambda regularization parameter (L2)
parser.add_argument('--lambda_param', type=float, default=0.0005,
                 help='L2 regularization parameter')
# cuda parameter
parser.add_argument('--use_cuda', action="store_true", default=False,
                 help='Use GPU or not')
# vanilla_lstm parameter
parser.add_argument('--vanilla_lstm', action="store_true", default=False,
                 help='True : vanilla_lstm cell, False: vanilla_lstm cell')
# drive option
parser.add_argument('--drive', action="store_true", default=False,
                 help='Use Google drive or not')
# number of validation will be used
parser.add_argument('--num_validation', type=int, default=2,
                 help='Total number of validation dataset for validate accuracy')
# frequency of validation
parser.add_argument('--freq_validation', type=int, default=1,
                 help='Frequency number(epoch) of validation using validation data')
# frequency of optimizer learning decay
parser.add_argument('--freq_optimizer', type=int, default=8,
                 help='Frequency number(epoch) of learning decay for optimizer')
# store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
parser.add_argument('--grid', action="store_true", default=True,
                 help='Whether store grids and use further epoch')

# dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

args = parser.parse_args()

cur_dataset = "eth"  #args.dataset_name

data_dir = os.path.join('/home/roongtaaahsih/ped_trajectory_prediction/sgan_ab/scripts/datasets/', cur_dataset + '/train')

''' Class for defining the vanilla_lstm Network '''
class VanillaLSTMNet(nn.Module):
    def __init__(self):
        super(VanillaLSTMNet, self).__init__()
        
        ''' Inputs to the vanilla_lstmCell's are (input, (h_0, c_0)):
         1. input of shape (batch, input_size): tensor containing input 
         features
         2a. h_0 of shape (batch, hidden_size): tensor containing the 
         initial hidden state for each element in the batch.
         2b. c_0 of shape (batch, hidden_size): tensor containing the 
         initial cell state for each element in the batch.
        
         Outputs: h_1, c_1
         1. h_1 of shape (batch, hidden_size): tensor containing the next 
         hidden state for each element in the batch
         2. c_1 of shape (batch, hidden_size): tensor containing the next 
         cell state for each element in the batch '''
        
        # set parameters for network architecture
        self.embedding_size = 64
        self.rnn_size=128
        self.input_size = 2
        self.output_size = 2
        self.dropout_prob = 0.5 
        
        # linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # define vanilla_lstm cell
        self.lstm_cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        # linear layer to map the hidden state of vanilla_lstm to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        
        pass
 
    def forward(self, observed_step,ht,ct):
    	peds,coords=observed_step.shape
    	lin_out=self.input_embedding_layer(observed_step.view(peds,2))
    	input_embedded=self.dropout(self.relu(lin_out))
    	ht,ct=self.lstm_cell(lin_out,(ht,ct))
    	out=self.output_layer(ht)
    	return out,ht,ct


	# '''define parameters for training and testing loops!'''

    # num_epoch = 20
    # pred_len = 12
    # learning_rate = 0.001
    # load trained model, if applicable
# if (cur_dataset != "eth"):
#     print("loading data from {}...".format(cur_dataset))
#     vanilla_lstm_net = torch.load('./saved_models/vanilla_lstm_model_' + cur_dataset + '_lr_0025_epoch_100_predlen_12.pt')
#     vanilla_lstm_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference
def train(vanilla_lstm_net,args):

	num_epoch = args.num_epochs
	pred_len = args.pred_len
	learning_rate = args.learning_rate
    
    # retrieve dataloader
	dataset, dataloader = loader.data_loader(args, data_dir)

	''' define the network, optimizer and criterion '''
	name="estep_1Opt" # to add to the name of files
	# if (cur_dataset == "eth"):
	# vanilla_lstm_net = VanillaLSTMNet()

	criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
	optimizer = optim.Adam(vanilla_lstm_net.parameters(), lr=learning_rate)

	# initialize lists for capturing losses/errors
	train_loss = []
	test_loss = []
	avg_train_loss = []
	avg_test_loss = []
	train_avgD_error=[]
	train_finalD_error=[]
	avg_train_avgD_error=[]
	avg_train_finalD_error=[]
	test_finalD_error=[]
	test_avgD_error=[]
	std_train_loss = []
	std_test_loss = []

	for i in range(num_epoch):
		print('======================= Epoch: {cur_epoch} / {total_epochs} =======================\n'.format(cur_epoch=i, total_epochs=num_epoch))
		for i, batch in enumerate(dataloader):
			train_batch = batch[0]
			target_batch = batch[1]
			# print("train_batch's shape", train_batch.shape)
			# print("target_batch's shape", target_batch.shape)
			seq, peds, coords = train_batch.shape # q is number of pedestrians 
			output_seq=[]
			ht = torch.zeros(train_batch.size(1), vanilla_lstm_net.rnn_size, dtype=torch.float)
			ct = torch.zeros(train_batch.size(1), vanilla_lstm_net.rnn_size, dtype=torch.float)
			for step in range(seq):
				out,ht,ct = vanilla_lstm_net(train_batch[step,:,:],ht,ct) # forward pass of vanilla_lstm network for training
			# print("out's shape:", out.shape)
			for step in range(pred_len):
				out,ht,ct=vanilla_lstm_net(out,ht,ct)
				output_seq+=[out]
				optimizer.zero_grad()
				step_loss=criterion(out,target_batch[step,:,:])
				step_loss.backward(retain_graph=True) #backward prop
				optimizer.step() #updating weights after each step prediction

        
			optimizer.zero_grad() # zero out gradients
			# print("output: {}".format(output_seq))
			output_seq = torch.stack(output_seq).squeeze() # convert list to tensor
			cur_train_loss = criterion(output_seq, target_batch) # calculate MSE loss
			# print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss
			# print('Current training loss: {}'.format(cur_train_loss.item())) # print current training loss

			#calculating average deisplacement error
			out1=output_seq
			target_batch1=target_batch  #making a copy of the tensors to convert them to array
			avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
			    np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
			train_avgD_error.append(avgD_error)

			#calculate final displacement error
			finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
			    np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
			train_finalD_error.append(finalD_error)

			train_loss.append(cur_train_loss.item())
			cur_train_loss.backward() # backward prop
			# optimizer.step() # step like a mini-batch (after all pedestrians)
		# optimizer.step() # update weights

        # save model at every epoch (uncomment) 
        # torch.save(vanilla_lstm_net, './saved_models/vanilla_lstm_model_v3.pt')
        # print("Saved vanilla_lstm_net!")
		avg_train_loss.append(np.sum(train_loss)/len(train_loss))
		avg_train_avgD_error.append(np.sum(train_avgD_error)/len(train_avgD_error))
		avg_train_finalD_error.append(np.sum(train_finalD_error)/len(train_finalD_error))   
		std_train_loss.append(np.std(np.asarray(train_loss)))
		train_loss = [] # empty train loss

		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		print("average train loss: {}".format(avg_train_loss[-1]))
		print("average std loss: {}".format(std_train_loss[-1]))
		avgTestLoss,avgD_test,finalD_test=test(vanilla_lstm_net,args,pred_len)
		avg_test_loss.append(avgTestLoss)
		test_finalD_error.append(finalD_test)
		test_avgD_error.append(avgD_test)
		print("test finalD error: ",finalD_test)
		print("test avgD error: ",avgD_test)
		#avg_test_loss.append(test(vanilla_lstm_net,args,pred_len)) ##calliing test function to return avg test loss at each epoch


    # '''after running through epochs, save your model and visualize.
    #    then, write your average losses and standard deviations of 
    #    losses to a text file for record keeping.'''

	save_path = os.path.join('./saved_models/', 'vanilla_lstm_model_'+name+'_lr_' + str(learning_rate) + '_epoch_' + str(num_epoch) + '_predlen_' + str(pred_len) + '.pt')
	# torch.save(vanilla_lstm_net, './saved_models/vanilla_lstm_model_lr001_ep20.pt')
	torch.save(vanilla_lstm_net, save_path)
	print("saved vanilla_lstm_net! location: " + save_path)

	''' visualize losses vs. epoch'''
	plt.figure() # new figure
	plt.title("Average train loss vs {} epochs".format(num_epoch))
	plt.plot(avg_train_loss,label='avg train_loss') 
	plt.plot(avg_test_loss,color='red',label='avg test_loss')
	plt.legend()
	plt.savefig("./saved_figs/" + "vanilla_lstm_"+name+"_avgtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) + '.jpeg')
	# plt.show()
	# plt.show(block=True)

	plt.figure() # new figure
	plt.title("Average and final displacement error {} epochs".format(num_epoch))
	plt.plot(avg_train_finalD_error,label='train:final disp. error') 
	plt.plot(avg_train_avgD_error,color='red',label='train:avg disp. error')
	plt.plot(test_finalD_error,color='green',label='test:final disp. error')
	plt.plot(test_avgD_error,color='black',label='test:avg disp. error')
	plt.ylim((0,10))
	plt.legend()
	# plt.show()
	plt.savefig("./saved_figs/" + "vanilla_lstm_"+name+"_avg_final_displacement_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) + '.jpeg')

	plt.figure()
	plt.title("Std of train loss vs epoch{} epochs".format(num_epoch))
	plt.plot(std_train_loss)
	plt.savefig("./saved_figs/" + "vanilla_lstm_"+name+"_stdtrainloss_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) + '.jpeg')
	# plt.show(block=True)
	print("saved images for avg training losses! location: " + "./saved_figs")

	# save results to text file
	txtfilename = os.path.join("./txtfiles/", "vanilla_lstm_"+name+"_avgtrainlosses_lr_"+ str(learning_rate) + '_epochs_' + str(num_epoch) + '_predlen_' + str(pred_len) + ".txt")
	os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
	with open(txtfilename, "w") as f:
		f.write("\n==============Average train loss vs. epoch:===============\n")
		f.write(str(avg_train_loss))
		f.write("\nepochs: " + str(num_epoch))
		f.write("\n==============Std train loss vs. epoch:===================\n")
		f.write(str(std_train_loss))
		f.write("\n==============Avg test loss vs. epoch:===================\n")
		f.write(str(avg_test_loss))
		f.write("\n==============Avg train displacement error:===================\n")
		f.write(str(avg_train_avgD_error))
		f.write("\n==============Final train displacement error:===================\n")
		f.write(str(avg_train_finalD_error))
		f.write("\n==============Avg test displacement error:===================\n")
		f.write(str(test_avgD_error))
		f.write("\n==============Final test displacement error:===================\n")
		f.write(str(test_finalD_error))
	print("saved average and std of training losses to text file in: ./txtfiles")

# test function to calculate and return avg test loss after each epoch
def test(vanilla_lstm_net,args,pred_len):


	test_data_dir = os.path.join('/home/roongtaaahsih/ped_trajectory_prediction/sgan_ab/scripts/datasets/', cur_dataset + '/test')

	# retrieve dataloader
	dataset, dataloader = loader.data_loader(args, test_data_dir)

	# define parameters for training and testing loops
	criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

	# initialize lists for capturing losses
	test_loss = []
	test_avgD_error=[]
	test_finalD_error=[]

	# now, test the model
	for i, batch in enumerate(dataloader):
		test_observed_batch = batch[0]
		test_target_batch = batch[1]
		# out = vanilla_lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of vanilla_lstm network for training
		seq, peds, coords = test_observed_batch.shape # q is number of pedestrians 
		output_seq=[]
		ht = torch.zeros(test_observed_batch.size(1), vanilla_lstm_net.rnn_size, dtype=torch.float)
		ct = torch.zeros(test_observed_batch.size(1), vanilla_lstm_net.rnn_size, dtype=torch.float)
		for step in range(seq):
			out,ht,ct = vanilla_lstm_net(test_observed_batch[step,:,:],ht,ct) # forward pass of vanilla_lstm network for training
		# print("out's shape:", out.shape)
		for step in range(pred_len):
			out,ht,ct=vanilla_lstm_net(out,ht,ct)
			output_seq+=[out]
		output_seq = torch.stack(output_seq).squeeze() # convert list to tensor
		cur_test_loss = criterion(output_seq, test_target_batch) # calculate MSE loss
		test_loss.append(cur_test_loss.item())
		out1=output_seq
		target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
		seq, peds, coords = test_target_batch.shape
		avgD_error=(np.sum(np.sqrt(np.square(out1[:,:,0].detach().numpy()-target_batch1[:,:,0].detach().numpy())+
            np.square(out1[:,:,1].detach().numpy()-target_batch1[:,:,1].detach().numpy()))))/(pred_len*peds)
		test_avgD_error.append(avgD_error)

		# final displacement error
		finalD_error=(np.sum(np.sqrt(np.square(out1[pred_len-1,:,0].detach().numpy()-target_batch1[pred_len-1,:,0].detach().numpy())+
            np.square(out1[pred_len-1,:,1].detach().numpy()-target_batch1[pred_len-1,:,1].detach().numpy()))))/peds
		test_finalD_error.append(finalD_error)
	avg_testloss = sum(test_loss)/len(test_loss)
	avg_testD_error=sum(test_avgD_error)/len(test_avgD_error)
	avg_testfinalD_error=sum(test_finalD_error)/len(test_finalD_error)
	print("============= Average test loss:", avg_testloss, "====================")

	return avg_testloss, avg_testD_error,avg_testfinalD_error



def main(args):
	vanilla_lstm_net = VanillaLSTMNet()
	train(vanilla_lstm_net,args)
	pass

if __name__ == '__main__':
    main(args)

