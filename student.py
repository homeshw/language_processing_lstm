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
	
# stop words not used due to the degraded performance. Commented for review
'''
stopWords = {'never','thru', 'under', 'latter', 'they', 'take', 'he', 'do', 'up', 'those', 'very', 'whereas', 'anyway', 'not', 'mostly', 'used', 'only', 'some', 'again', "'s", 'somewhere', 'we', 'between', 'sixty', 'seem', 'yourselves', 'why', 'nor', 'behind', 'nine', 'per', 'yours', 'three', 'less', 'please', 'otherwise', "'ve", 'even', "'m", 'same', 'yourself', 'either', 'these', 'always', 'other', 'done', 'side', 'back', 'doing', 'his', 'at', 'once', 'have', 'regarding', 'somehow', 'sometime', 'too', 'was', 'with', 'but', 'me', 're', 'quite', 'twenty', 'a', 'most', 'their', 'here', 'say', 'on', 'well', 'anything', 'empty', 'what', 'rather', 'my', 'also', 'is', 'via', 'herself', 'various', 'him', 'may', 'both', 'should', 'anyhow', 'just', 'it', 'four', 'least', 'beforehand', 'something', 'whoever', 'whether', 'itself', 'in', 'nevertheless', 'except', 'beyond', 'your', "'d", 'name', 'upon', 'already', 'neither', 'often', 'below', 'which', 'wherever', 'becomes', 'everywhere', 'put', 'none', 'after', 'almost', 'five', 'therefore', 'twelve', 'whole', 'eleven', 'off', 'the', 'down', 'by', 'along', 'how', 'whatever', 'noone', 'formerly', 'while', 'much', 'into', 'six', 'former', 'namely', 'himself', 'there', 'ours', 'last', 'unless', 'onto', 'than', 'them', 'being', 'third', 'amongst', 'seems', 'really', 'within', 'because', 'show', 'however', 'move', 'further', 'us', 'ever', 'became', 'front', 'becoming', 'of', 'several', 'be', 'such', 'hers', 'though', 'each', 'whom', 'seeming', 'whose', 'i', 'until', 'forty', 'now', 'against', 'ten', 'throughout', 'anywhere', 'become', 'can', 'our', 'about', 'get', 'alone', 'few', 'no', 'due', 'themselves', 'had', 'although', 'nowhere', 'top', 'part', 'beside', 'many', 'her', 'its', 'whereby', 'as', 'since', "'ll", 'anyone', 'eight', 'hereby', 'still', 'indeed', 'first', 'more', 'hence', 'from', 'out', 'whenever', 'towards', 'everything', 'one', 
'must', 'ourselves', 'make', 'or', 'hundred', 'did', "'re", 'serious', 'among', 'thereafter', 'besides', 'meanwhile', 'where', 'nobody', 'others', 'next', 'before', 'fifteen', 'if', 'fifty', 'another', 'over', 'mine', 'two', 'when', 'without', 'to', 'could', 'elsewhere', 'full', 'together', 'then', 'sometimes', 'thus', 'keep', 'own', 'who', 'around', 'you', 'an', 'been', 'enough', 'nothing', 'so', 'everyone', 'afterwards', 'across', 'this', 'see', 'yet', 'perhaps', "n't", 'myself', 'made', 'any', 'through', 'else', 'bottom', 'would', 'ca', 'toward', 'seemed', 'will', 'someone', 'all', 'every', 'using', 'during', 'and', 'for', 'has', 'she', 'above', 'give', 'go', 'were', 'amount', 'does', 'call', 'are', 'that', 'am', 'might'}
'''
stopWords = {}

word_vec_dimension = 200
wordVectors = GloVe(name='6B', dim=word_vec_dimension)

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
    # rounding. Therefore values > 0.5 will be categorized as 1 and 0 otherwise
    ratingOutputPred = torch.round(ratingOutput)

    # convert the output value to a value between 0-5
    categoryOutputPred = categoryOutput.argmax(dim=1, keepdim=True)

    return ratingOutputPred, categoryOutputPred

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

    def __init__(self,batch_size,input_dim,hidden_nodes,output_dim,number_of_layers):
        super(network, self).__init__()
		
        # parameter initialization
        self.batch_size = batch_size
        self.num_hid = hidden_nodes
        self.num_layers = number_of_layers
		
        # layer definitions
        self.W = tnn.Parameter(torch.Tensor(input_dim, self.num_hid * 4))
        self.U = tnn.Parameter(torch.Tensor(self.num_hid, self.num_hid * 4))
        self.hid_bias = tnn.Parameter(torch.Tensor(self.num_hid * 4))
        self.V = tnn.Parameter(torch.Tensor(self.num_hid, output_dim))
        self.out_bias = tnn.Parameter(torch.Tensor(output_dim))
		
        # layer initializations
        self.init_weights()
        self.init_hidden()
		
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, input, length):
        # input is of shape (batch, sequence, feature)
        init_states = None
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
        size = output.size()
        I = size[0]
        K = size[2]
		
		# taking output considering the zero padding
        fOutput = torch.zeros(I,K).to(input.device)
        for i in range(I):
            for k in range(K):
                j = length[i]-1
                fOutput[i][k] = output[i][j][k]
		# activate rating output nodes
        ratingOutputA = torch.sigmoid(fOutput[:,0])
		# activate category output nodes
        categoryOutputA = F.log_softmax(fOutput[:,1:6],dim=1)
        return ratingOutputA,categoryOutputA

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

        # calculating loss for outputs
        loss_rating = self.loss_function_binary(ratingOutput.squeeze(), ratingTarget.float().squeeze())
        loss_category = self.loss_function_multi(categoryOutput.squeeze(), categoryTarget.squeeze())
		
		# loss functions are added to get the total loss function
        total_loss = loss_rating + loss_category
		
        return total_loss

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.7
batchSize = 32
epochs = 8

input_dim = word_vec_dimension
hidden_nodes = 100
output_dim = 6
number_of_layers = 1

net = network(batchSize,input_dim,hidden_nodes,output_dim,number_of_layers)
optimiser = toptim.Adam(net.parameters(),lr=0.001)
lossFunc = loss()
