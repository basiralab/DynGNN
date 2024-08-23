import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_geometric.nn import NNConv
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.nn import Linear, Sequential, ReLU
from data_utils import create_edge_index_attribute
from torch.nn.parameter import Parameter
from torch import mm as mm
from torch.nn import Tanh
from models.echo import EchoStateNetwork
import torch.nn.functional as F
import numpy as np
import random
import os

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
else:
    device = torch.device("cpu")
    print('running on CPU')


shape = torch.Size((1225, 1225))
hidden_state = torch.cuda.FloatTensor(shape)
torch.randn(shape, out=hidden_state)


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=True)
        self.weight_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tanh = Tanh()
        shape = torch.Size((1225, 1225))
        self.hidden_state = torch.FloatTensor(shape).to(device)
        torch.randn(shape, out=hidden_state)
        self.hidden_state = torch.nn.functional.normalize(hidden_state)

    def forward(self, x):
        # global hidden_state
        h = self.hidden_state
        y = self.tanh(self.weight(x) + self.weight_h(h))
        self.hidden_state = y.detach()
        return y

    def update_h(self, hidden):
        self.hidden_state = hidden

    def get_h(self):
        return self.hidden_state
    
def eucledian_distance(x):
    repeated_out = x.repeat(35, 1, 1)
    repeated_t = torch.transpose(repeated_out, 0, 1)
    diff = torch.abs(repeated_out - repeated_t)
    return torch.sum(diff, 2)
    

class GNN_1(nn.Module):
    def __init__(self, device=device, input_weights=None):
        super(GNN_1, self).__init__()
        self.rnn = nn.Sequential(RNNCell(1, 1225), ReLU())
        self.gnn_conv = NNConv(35, 35, self.rnn, aggr='mean', root_weight=True, bias=True)
        self.esn = EchoStateNetwork(device=device, input_weights=input_weights)
        self.linear = nn.Linear(36, 20).to(dtype=torch.float64)

    def forward(self, data, train_sig_input=None, train_sig_output=None, test_sig_input=None):
        edge_index, edge_attr, _, _ = create_edge_index_attribute(data)
        x1 = F.relu(self.gnn_conv(data, edge_index.to(device), edge_attr.to(device)))
        #x1 = eucledian_distance(x1)
        if train_sig_input is not None:
            hidden_state = self.esn(x1, train_sig_input)
            self.esn.train_output_weights(hidden_state, train_sig_output)
            output_sig = self.esn.predict(test_sig_input)
            return x1, output_sig
        else:
            return x1

    def update_h(self, hidden):
        self.rnn.update_h(hidden)

    def get_h(self):
        return self.rnn.get_h()


def frobenious_distance(test_sample, predicted):
    diff = torch.abs(test_sample - predicted)
    dif = diff * diff
    sum_of_all = diff.sum()
    d = torch.sqrt(sum_of_all)
    return d
