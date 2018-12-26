#Script to load GRU model and test
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
from gru_prototype_v1 import GRUNet # class definition needed

# build argparser
parser = argparse.ArgumentParser()

parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--rnn_size', type=int, default=128,
                 help='size of RNN hidden state')
# Size of each batch parameter
parser.add_argument('--batch_size', type=int, default=10,
                 help='minibatch size')
# Length of sequence to be considered parameter
parser.add_argument('--seq_length', type=int, default=20,
                 help='RNN sequence length')
parser.add_argument('--pred_length', type=int, default=12,
                 help='prediction length')
# Number of epochs parameter
parser.add_argument('--num_epochs', type=int, default=30,
                 help='number of epochs')
# Frequency at which the model should be saved parameter
parser.add_argument('--save_every', type=int, default=400,
                 help='save frequency')
# TODO: (resolve) Clipping gradients for now. No idea whether we should
# Gradient value at which it should be clipped
parser.add_argument('--grad_clip', type=float, default=10.,
                 help='clip gradients at this value')
# Learning rate parameter
parser.add_argument('--learning_rate', type=float, default=0.003,
                 help='learning rate')
# Decay rate for the learning rate parameter
parser.add_argument('--decay_rate', type=float, default=0.95,
                 help='decay rate for rmsprop')
# Dropout not implemented.
# Dropout probability parameter
parser.add_argument('--dropout', type=float, default=0.5,
                 help='dropout probability')
# Dimension of the embeddings parameter
parser.add_argument('--embedding_size', type=int, default=64,
                 help='Embedding dimension for the spatial coordinates')
# Size of neighborhood to be considered parameter
parser.add_argument('--neighborhood_size', type=int, default=32,
                 help='Neighborhood size to be considered for social grid')
# Size of the social grid parameter
parser.add_argument('--grid_size', type=int, default=4,
                 help='Grid size of the social grid')
# Maximum number of pedestrians to be considered
parser.add_argument('--maxNumPeds', type=int, default=27,
                 help='Maximum Number of Pedestrians')

# Lambda regularization parameter (L2)
parser.add_argument('--lambda_param', type=float, default=0.0005,
                 help='L2 regularization parameter')
# Cuda parameter
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
# frequency of optimazer learning decay
parser.add_argument('--freq_optimizer', type=int, default=8,
                 help='Frequency number(epoch) of learning decay for optimizer')
# store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
parser.add_argument('--grid', action="store_true", default=True,
                 help='Whether store grids and use further epoch')

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)

args = parser.parse_args()

# load trained model
gru_net = torch.load('./saved_models/GRU_lr0005_ep10.pt')
gru_net.eval() # set dropout and batch normalization layers to evaluation mode before running inference

def test(args):
    test_data_dir = "/home/roongtaaahsih/ped_traj/sgan_ab/scripts/datasets/zara1/test"

    # retrieve dataloader
    dataset, dataloader = loader.data_loader(args, test_data_dir)

    # define parameters for training and testing loops
    pred_len = 12
    criterion = nn.MSELoss() # MSE works best for difference between predicted and actual coordinate paths

    # initialize lists for capturing losses
    test_loss = []

    # now, test the model
    for i, batch in enumerate(dataloader):
      test_observed_batch = batch[0]
      test_target_batch = batch[1]
      out = gru_net(test_observed_batch, pred_len=pred_len) # forward pass of gru network for training
      print("out's shape:", out.shape)
      cur_test_loss = criterion(out, test_target_batch) # calculate MSE loss
      print('Current test loss: {}'.format(cur_test_loss.item())) # print current test loss
      test_loss.append(cur_test_loss.item())
    avg_testloss = sum(test_loss)/len(test_loss)
    print("========== Average test loss:", avg_testloss, "==========")

    pass

def main():
    test(args) # test all of the data in test_data_dir
    print("Done testing!")

if __name__ == '__main__':
    main()