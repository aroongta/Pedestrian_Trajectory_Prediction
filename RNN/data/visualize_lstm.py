#Script to load model, predict sequences and plot
# import relevant libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
from lstm_prototype_v3 import VanillaLSTMNet # class definition needed
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
parser.add_argument('--seq_length', type=int, default=16,
                 help='RNN sequence length')
parser.add_argument('--pred_length', type=int, default=8,
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
# lstm parameter
parser.add_argument('--lstm', action="store_true", default=False,
                 help='True : lstm cell, False: LSTM cell')
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
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

args = parser.parse_args()

cur_dataset = 'zara1' #args.dataset_name

data_dir = os.path.join('/home/roongtaaahsih/ped_traj/sgan_ab/scripts/datasets/', cur_dataset + '/test')
# load trained model
lstm_net = torch.load('./saved_models/vanilla_lstm_model_lr_0.0017_epoch_100_predlen_8_batchsize_5.pt')
lstm_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference
# test function to calculate and return avg test loss after each epoch
def test(lstm_net,args,pred_len,data_dir):

    test_data_dir = data_dir #os.path.join('/home/ashishpc/Desktop/sgan_ab/scripts/datasets/', cur_dataset + '/train')

    # retrieve dataloader
    dataset, dataloader = loader.data_loader(args, test_data_dir)

    # define parameters for training and testing loops
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []
    test_avgD_error=[]
    test_finalD_error=[]
    # obs=[]
    # predict=[]
    # ground_truth=[]
    plt.figure(figsize=(32,20))
    plt.xlabel("X coordinates of pedestrians")
    plt.ylabel("Y coordinates of pedestrians")
    # now, test the model
    for i, batch in enumerate(dataloader):
        test_observed_batch = batch[0]
        test_target_batch = batch[1]
        out = lstm_net(test_observed_batch, pred_len=pred_len) # forward pass of lstm network for training
        # cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
        # test_loss.append(cur_test_loss.item())
        s,peds,c=out.shape
        out1=out.detach().numpy()
        target1=test_target_batch.detach().numpy()
        observed1=test_observed_batch.detach().numpy()
        print("observed 1 shape:",observed1.shape)
        print("target1 shape:", target1.shape)
        print("out 1 shape", out1.shape)
        out2=np.vstack((observed1,out1))
        target2=np.vstack((observed1,target1))
        print("out2 shape",out2.shape)
        for t in range(6):
            plt.plot(observed1[:,t,0],observed1[:,t,1],color='b',marker='o',linewidth=5,markersize=12)
            plt.plot(target2[s-1:s+pred_len,t,0],target2[s-1:s+pred_len,t,1],color='red',marker='o',linewidth=5,markersize=12)
            plt.plot(out2[s-1:s+pred_len,t,0],out2[s-1:s+pred_len,t,1],color='g',marker='o',linewidth=5,markersize=12)
        plt.legend(["Observed","Ground Truth","Predicted"])
        plt.show(block=True)
    

        # out1=out
        # target_batch1=test_target_batch  #making a copy of the tensors to convert them to array
        # seq, peds, coords = test_target_batch.shape

    return test_observed_batch,test_target_batch,out

def main(args):
    
    '''define parameters for training and testing loops!'''

    # num_epoch = 20
    # pred_len = 12
    # learning_rate = 0.001

    num_epoch = 2 #args.num_epochs
    pred_len =8 #args.pred_len
    learning_rate = 0.0017 #args.learning_rate
    
    # retrieve dataloader
    dataset, dataloader = loader.data_loader(args, data_dir)

    ''' define the network, optimizer and criterion '''
    # lstm_net = lstmNet()

    # criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths
    # # optimizer = optim.Adam(lstm_net.parameters(), lr=learning_rate)

    # # initialize lists for capturing losses/errors
    # test_loss = []
    # avg_test_loss = []
    # test_finalD_error=[]
    # test_avgD_error=[]
    # std_test_loss = []

    #calling the test function and calculating the test losses
    test_observed_batch,test_target_batch,out=test(lstm_net,args,pred_len,data_dir)
    pass

'''main function'''
if __name__ == '__main__':
    main(args)