#!/usr/bin/env python3
"""
student.py

UNSW ZZEN9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as
a basic tokenise function.  You are encouraged to modify these to improve
the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import math
# import numpy as np
# import sklearn
#import sys

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # fix. Find better way if possible
    ratingOutputPred = torch.round(ratingOutput)

    # convert to 0-5
    categoryOutputPred = categoryOutput.argmax(dim=1, keepdim=True)

    return ratingOutput, categoryOutputPred

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.batch_size = 32
        num_input = 50 # fix
        self.num_hid = 7
        num_out = 6
        self.num_layers = 1
        self.W = tnn.Parameter(torch.Tensor(num_input, self.num_hid * 4))
        self.U = tnn.Parameter(torch.Tensor(self.num_hid, self.num_hid * 4))
        self.hid_bias = tnn.Parameter(torch.Tensor(self.num_hid * 4))
        self.V = tnn.Parameter(torch.Tensor(self.num_hid, num_out))
        self.out_bias = tnn.Parameter(torch.Tensor(num_out))
        self.init_weights()
        self.init_hidden() #check
		
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, input, length):
        """Assumes x is of shape (batch, sequence, feature)"""
        init_states = None
        #with open('input.txt', 'w') as f:
        #    f.write(str(input))
        batch_size, seq_size, _ = input.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid).to(input.device), 
                        torch.zeros(batch_size,self.num_hid).to(input.device))
        else:
            h_t, c_t = init_states
         
        NH = self.num_hid
        for t in range(seq_size):
            x_t = input[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :NH]),     # input gate
                torch.sigmoid(gates[:, NH:NH*2]), # forget gate
                torch.tanh(gates[:, NH*2:NH*3]),  # new values
                torch.sigmoid(gates[:, NH*3:]),   # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        #with open('output.txt', 'w') as f:
        #    f.write(str(output))
        ratingOutput = F.sigmoid(output[:,-1,0])
        categoryOutput = F.softmax(output[:,-1,1:6])
        #categoryOutput = F.log_softmax(output[:,-1,1:6], dim=1)
        #categoryOutput = output[:,:,1:6]
        #return ratingOutput,categoryOutput
        return ratingOutput,categoryOutput

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss_function_binary = F.binary_cross_entropy
        self.loss_function_multi = F.nll_loss
		

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):

        # convert categoryTarget into one-hot encoding
        one_hot_categoryTarget = F.one_hot(categoryTarget)

        #prob_rating  = F.sigmoid(ratingOutput)		
        #log_prob_category = F.log_softmax(categoryOutput, dim=1)
        #prob_category = F.softmax(categoryOutput)

        #category_output_flatten = torch.flatten(categoryOutput,start_dim=0)
        #category_target_flatten = torch.flatten(one_hot_categoryTarget,start_dim=0)
        #category_output_flatten = categoryOutput.view(160)
        #category_target_flatten = one_hot_categoryTarget.view(160)
        #categoryOutput = torch.transpose(categoryOutput, 0, 1)
        #one_hot_categoryTarget = torch.transpose(one_hot_categoryTarget, 0, 1)
		
        loss_rating = self.loss_function_binary(ratingOutput.squeeze(), ratingTarget.float().squeeze())
        loss_category = self.loss_function_multi(categoryOutput.squeeze(), categoryTarget.squeeze())
        #loss_category = self.loss_function_multi(category_output_flatten.squeeze(), category_target_flatten.squeeze())
        #loss_category = self.loss_function_multi(categoryOutput.squeeze(), one_hot_categoryTarget.squeeze())
        #loss_category = 0
		
        total_loss = loss_rating + loss_category
		
        return total_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
#fix
epochs = 1
optimiser = toptim.SGD(net.parameters(), lr=0.01)
