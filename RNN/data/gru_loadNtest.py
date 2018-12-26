#Script to load GRU model trained on all datasets and test
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
from gru_prototype_v4_alldata import GRUNet # class definition needed
# build argparser
parser = argparse.ArgumentParser()

parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=2)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                 help='size of RNN hidden state')
# size of each batch parameter
parser.add_argument('--batch_size', type=int, default=10,
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
# GRU parameter
parser.add_argument('--gru', action="store_true", default=False,
                 help='True : GRU cell, False: LSTM cell')
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
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

args = parser.parse_args()

cur_dataset = args.dataset_name

data_dir = os.path.join('/home/roongtaaahsih/ped_traj/sgan_ab/scripts/datasets/', cur_dataset + '/test')
# load trained model
gru_net = torch.load('./saved_models/gru_model_zara2_lr_0.0025_epoch_100_predlen_12.pt')
gru_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference
# test function to calculate and return avg test loss after each epoch
def test(gru_net,args,pred_len,data_dir):

    test_data_dir = data_dir #os.path.join('/home/ashishpc/Desktop/sgan_ab/scripts/datasets/', cur_dataset + '/train')

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
        out = gru_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        test_loss.append(cur_test_loss.item())
        out1=out
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
    
    '''define parameters for training and testing loops!'''

    # num_epoch = 20
    # pred_len = 12
    # learning_rate = 0.001

    num_epoch = args.num_epochs
    pred_len = args.pred_len
    learning_rate = args.learning_rate
    
    # retrieve dataloader
    dataset, dataloader = loader.data_loader(args, data_dir)

    ''' define the network, optimizer and criterion '''
    # gru_net = GRUNet()

    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    # optimizer = optim.Adam(gru_net.parameters(), lr=learning_rate)

    # # initialize lists for capturing losses/errors
    # test_loss = []
    # avg_test_loss = []
    # test_finalD_error=[]
    # test_avgD_error=[]
    # std_test_loss = []

    #calling the test function and calculating the test losses
    avg_test_loss,test_avgD_error,test_finalD_error=test(gru_net,args,pred_len,data_dir)
    
    # save results to text file
    txtfilename = os.path.join("./txtfiles/", r"Trained_all_Results_table_lr_"+ str(learning_rate)+ ".txt")
    os.makedirs(os.path.dirname("./txtfiles/"), exist_ok=True) # make directory if it doesn't exist
    with open(txtfilename, "a+") as f: #will append to a file, create a new one if it doesn't exist
        if(pred_len==2): #To print the heading in the txt file
        	f.write("Pred_Len"+"\t"+"Avg_Test_Loss"+"\t"+"Test_AvgD_Error"+"\t"+"Test_FinalDisp_Error"+"\n") 
        #f.write("\n==============Average train loss vs. epoch:===============")
        f.write(str(pred_len)+"\t")
        #f.write("\n==============avg test loss vs. epoch:===================")
        f.write(str(avg_test_loss)+"\t")
        #f.write("\n==============Avg test displacement error:===================")
        f.write(str(test_avgD_error)+"\t")
        #f.write("\n==============final test displacement error:===================")
        f.write(str(test_finalD_error)+"\n")
        f.close()
    print("saved average and std of training losses to text file in: ./txtfiles")
'''main function'''
if __name__ == '__main__':
	main(args)