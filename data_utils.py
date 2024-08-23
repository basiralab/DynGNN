import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.utils import dense_to_sparse

import networkx as nx


def create_edge_index_attribute(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = torch.zeros((2, rows * cols), dtype=torch.long)
    edge_attr = torch.zeros((rows * cols, 1), dtype=torch.float)
    counter = 0

    for src, attrs in enumerate(adj_matrix):
        for dest, attr in enumerate(attrs):
            edge_index[0][counter], edge_index[1][counter] = src, dest
            edge_attr[counter] = attr
            counter += 1

    return edge_index, edge_attr, rows, cols

def adj_matrix_to_pytorch_geometric_data(adj_matrix, device):

    # calculate edge_index and edge_weights
    edge_indices, edge_weights = dense_to_sparse(adj_matrix)

    # edge attributes
    edge_attr = torch.cat([edge_indices.T,edge_weights.view(len(edge_weights),1)],1)

    # calculate the node features
    x = node_features_from_adj_matrix(adj_matrix,device)

    data = Data(x=x.to(device),
                edge_index=edge_indices.to(device),
                edge_weights=edge_weights.to(device),
                adj_matrix=adj_matrix.to(device),
                edge_attr=edge_attr.to(device))
    return data

def node_features_from_adj_matrix(adj_matrix, device):

      if device.type == 'cpu':
          # Create a NetworkX graph from the adjacency matrix
          G = nx.from_numpy_array(adj_matrix.detach().numpy())
      elif device.type == 'cuda':
          # Create a NetworkX graph from the adjacency matrix
          G = nx.from_numpy_array(adj_matrix.detach().cpu().numpy())

      # Compute the weighted degree (strength) for each node
      strength = dict(G.degree(weight='weight'))

      # Compute the clustering coefficient for each node
      clustering = nx.clustering(G, weight='weight')

      # Compute the PageRank for each node
      #pagerank = nx.pagerank(G, weight='weight', max_iter=500)

      # Let's convert these features into numpy arrays so we can stack them together
      strength_array = np.array(list(strength.values()))
      clustering_array = np.array(list(clustering.values()))
      #pagerank_array = np.array(list(pagerank.values()))

      # Now we can stack these features together to get a node feature matrix
      x = torch.Tensor(np.vstack([strength_array, clustering_array]).T)
      return x
    
def swap(data):
    # Swaps the x & y values of the given graph
    edge_i, edge_attr, _, _ = create_edge_index_attribute(data.y)
    data_s = Data(x=data.y, edge_index=edge_i, edge_attr=edge_attr, y=data.x)
    return data_s