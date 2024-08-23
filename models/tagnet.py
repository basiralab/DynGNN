import torch
import torch.nn as nn
import torch.nn.functional as F
from models.echo import EchoStateNetwork
from torch_geometric.nn.conv import TAGConv
from data_utils import adj_matrix_to_pytorch_geometric_data


class DeepTAGNet(torch.nn.Module):
    def __init__(self, device, input_weights=None):
        super(DeepTAGNet, self).__init__()
        in_features = 35
        out_features = 140
        self.conv1 = TAGConv(in_features, out_features)
        self.conv2 = TAGConv(out_features, 2*out_features)
        self.fc1 = nn.Linear(2*out_features,35)
        self.device = device
        self.esn = EchoStateNetwork(device=device, input_weights=input_weights)

    def forward(self, adj_matrix, train_sig_input=None, train_sig_output=None, test_sig_input=None):

        data = adj_matrix_to_pytorch_geometric_data(adj_matrix,self.device)
        edge_index = data.edge_index
        data = self.conv1(data.adj_matrix, edge_index)
        data = F.elu(data)
        data = F.dropout(data, p=0.3, training=self.training)

        data = self.conv2(data, edge_index)
        data = F.elu(data)
        data = F.dropout(data, p=0.5, training=self.training)

        out = F.relu(self.fc1(data))
        
        if train_sig_input is not None:
            hidden_state = self.esn(out, train_sig_input)
            self.esn.train_output_weights(hidden_state, train_sig_output)
            output_sig = self.esn.predict(test_sig_input)
            return out, output_sig
        else:
            return out