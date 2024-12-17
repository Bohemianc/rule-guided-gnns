import torch
import torch.nn.functional as F
from torch.nn import Parameter, Module

from torch_geometric.nn import MessagePassing


class EC_GCNConv(MessagePassing):
    def __init__(self, num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr='add'):
        super(EC_GCNConv, self).__init__(aggr=aggr)

        self.num_features = num_unary + num_binary
        self.A = Parameter(torch.Tensor(self.num_features, self.num_features))
        self.B = Parameter(torch.Tensor(num_edge_types, self.num_features, self.num_features))
        self.bias_single = Parameter(torch.Tensor(self.num_features))
        self.bias_pair = Parameter(torch.Tensor(self.num_features))
        # self.A.data.uniform_(0, 1)
        # self.B.data.uniform_(0, 1)
        self.A.data.normal_(0, 0.01)
        self.B.data.normal_(0, 0.01)
        self.bias_single.data.normal_(0, 0.001)
        self.bias_pair.data.normal_(0, 0.001)
        self.num_unary = num_unary
        self.num_binary = num_binary
        self.num_edge_types = num_edge_types
        self.num_singleton_nodes = num_singleton_nodes
        self.aggr = aggr

    def forward(self, x, edge_index, edge_type):
        if self.aggr == 'add':
            out = F.linear(x, self.A, bias=None)

        for i in range(self.num_edge_types):
            edge_mask = edge_type == i
            temp_edges = edge_index[:, edge_mask]
            msg = self.propagate(temp_edges, x=x, size=(x.size(0), x.size(0)))
            if self.aggr == 'add':
                msg = F.linear(msg, self.B[i], bias=None)
                out += msg

        out[:self.num_singleton_nodes] += self.bias_single
        out[self.num_singleton_nodes:] += self.bias_pair

        out = clipped_relu(out)

        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, num_unary, num_binary, num_edge_types, num_singleton_nodes=0, num_layers=2, dropout=0, aggr='add'):
        super(GNN, self).__init__()
        self.conv1 = EC_GCNConv(num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr=aggr)
        self.conv2 = EC_GCNConv(num_unary, num_binary, num_edge_types, num_singleton_nodes, aggr=aggr)
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.conv1(x, edge_index, edge_type)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)

        assert torch.max(x).item() <= 1.0

        return x


def modified_sigmoid(x, k=1, c=0):
    return 1 / (1 + torch.exp(-k * (x - c)))

def clipped_relu(x, a=1):
    return torch.clamp(x, 0, a)