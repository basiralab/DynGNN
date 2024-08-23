import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from models.echo import EchoStateNetwork
from data_utils import adj_matrix_to_pytorch_geometric_data
    
class GAT(torch.nn.Module):
    def __init__(self, device, input_weights=None):
        super(GAT, self).__init__()
        self.in_channels = 35
        self.hidden_channels = 120
        self.heads=16
        self.out_head = 1
        self.device = device
        self.num_nodes = 35

        self.conv1 = GATConv(self.in_channels, self.hidden_channels, heads=self.heads, dropout=0.6)
        self.conv2 = GATConv(self.hidden_channels*self.heads, self.hidden_channels, heads=self.heads, dropout=0.6)
        self.conv3 = GATConv(self.hidden_channels*self.heads, self.hidden_channels, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.fc = nn.Linear(self.hidden_channels,self.num_nodes)
        
        self.esn = EchoStateNetwork(device=device, input_weights=input_weights)

    def forward(self, adj_matrix, train_sig_input=None, train_sig_output=None, test_sig_input=None):

        data = adj_matrix_to_pytorch_geometric_data(adj_matrix, self.device)
        edge_index = data.edge_index

        x = F.dropout(data.adj_matrix, p=0.6, training=self.training)
        data = self.conv1(x, edge_index)
        data = F.elu(data)
        data = F.dropout(data, p=0.6, training=self.training)
        data = self.conv2(data, edge_index)
        data = F.elu(data)
        data = F.dropout(data, p=0.6, training=self.training)
        data = self.conv3(data, edge_index)
        out = F.relu(self.fc(data))

        if train_sig_input is not None:
            hidden_state = self.esn(out, train_sig_input)
            self.esn.train_output_weights(hidden_state, train_sig_output)
            output_sig = self.esn.predict(test_sig_input)
            return out, output_sig
        else:
            return out