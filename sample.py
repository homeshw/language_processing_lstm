import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import math
import random

# code adapted from
# http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/reberGrammar.php

# assign a number to each transition
chars='BTSXPVE'

# finite state machine for non-embedded Reber Grammar
graph = [[(1,5),('T','P')] , [(1,2),('S','X')], \
         [(3,5),('S','X')], [(6,),('E')], \
         [(3,2),('V','P')], [(4,5),('V','T')] ]

def get_one_example(min_length = 5):
    seq = [0]
    node = 0
    prob = []
    while node != 6:
        this_prob = np.zeros(7)
        transitions = graph[node]
        if (len(seq) < min_length - 2) and (node == 2 or node == 4):
            # choose transition to force a longer sequence
            i = 1
            this_prob[chars.find(transitions[1][1])] = 1 
        else:
            # choose transition randomly
            i = np.random.randint(0, len(transitions[0]))
            for ch in transitions[1]:
                this_prob[chars.find(ch)] = 1./len(transitions[1])
        prob.append(this_prob)
        seq.append(chars.find(transitions[1][i]))
        node = transitions[0][i]
    return seq, prob

def get_one_embedded_example(min_length=9):
    i = np.random.randint(0,2)  # choose between 'T' and 'P'
    if i == 0:
        first = 1 # 'T'
        prob1 = 1
        prob4 = 0
    else:
        first = 4 # 'P'
        prob1 = 0
        prob4 = 1
    seq_mid, prob_mid = get_one_example(min_length-4)
    seq = [0,first] + seq_mid  + [first,6]
    prob = [(0,0.5,0,0,0.5,0,0),(1,0,0,0,0,0,0)] + prob_mid + \
           [(0,prob1,0,0,prob4,0,0),(0,0,0,0,0,0,1)]
    return seq, prob

def get_reber_sequence(embedded=False,min_length=4):
    if embedded:
        seq, prob = get_one_embedded_example(min_length)
    else:
        seq, prob = get_one_example(min_length)

    # convert numpy array to torch tensor
    seq_torch = torch.from_numpy(np.asarray(seq))
    input = F.one_hot(seq_torch[0:-1],num_classes=7).float()
    label = seq_torch[1:]
    probs = torch.from_numpy(np.asarray(prob)).float()
    input = input.unsqueeze(0)
    label = label.unsqueeze(0)
    probs = probs.unsqueeze(0)
    return input, label, probs

class LSTM_model(nn.Module):
    def __init__(self,num_input,num_hid,num_out,batch_size=1,num_layers=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid).to(x.device), 
                        torch.zeros(batch_size,self.num_hid).to(x.device))
        else:
            h_t, c_t = init_states
         
        NH = self.num_hid
        for t in range(seq_size):
            x_t = x[:, t, :]
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
        return output


def train(model_type='srn', hid=8, embed=False, length=4, lr=0.3):
    model = LSTM_model(7,hid,7)

    loss_function = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=lr)

    np.set_printoptions(suppress=True,precision=2)

    for epoch in range(50001):
        model.zero_grad()
        input, label, prob = get_reber_sequence(embedded=embed,
                                                min_length=length)
        model.init_hidden()
        output = model(input)
        log_prob  = F.log_softmax(output, dim=2)
        loss = loss_function(log_prob.squeeze(), label.squeeze())
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            # Check accuracy during training
            with torch.no_grad():
                model.eval()
                input, label, prob = get_reber_sequence(embedded=embed,
                                                        min_length=length)
                model.init_hidden()
                output = model(input)
                log_prob  = F.log_softmax(output, dim=2)
                prob_out = torch.exp(log_prob)
                print('-----')
                symbol = [chars[index] for index in label.squeeze().tolist()]
                print('symbol = B'+''.join(symbol))
                print('label =',label.squeeze().numpy())
                print('true probabilities:')
                print(prob.squeeze().numpy())
                print('output probabilities:')
                print(prob_out.squeeze().numpy())
                print('epoch: %d' %epoch)
                if embed:
                    prob_out_mid   = prob_out[:,2:-3,:]
                    prob_out_final = prob_out[:,-2,:]
                    prob_mid   = prob[:,2:-3,:]
                    prob_final = prob[:,-2,:]
                    print('error: %1.4f' %torch.mean((prob_out_mid - prob_mid)
                                                    *(prob_out_mid - prob_mid)))
                    print('final: %1.4f' %torch.mean((prob_out_final - prob_final)
                                                    *(prob_out_final - prob_final)))
                else:
                    print('error: %1.4f' %torch.mean((prob_out - prob)
                                                    *(prob_out - prob)))
                model.train()

model='lstm'
hid=8
embed=True
length=4

train(model_type=model, hid=hid, embed=embed, length=4)