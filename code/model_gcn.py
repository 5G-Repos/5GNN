import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, global_mean_pool
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, DenseGINConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SuperGATConv, GINConv, GCN2Conv, EGConv


# GCN
class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        self.conv1 = GCNConv(self.config["input_dim"], 128)
        self.norm1 = LayerNorm(128)
        self.conv2 = GCNConv(128, 128)
        self.norm2 = LayerNorm(128)
        self.lin = nn.Linear(128 * (self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight, batch=None):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight)))
        # h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index, edge_weight)))
        # h2 = F.dropout(h2, training=self.training)
        if batch is not None:
            # [batch_size * nodes, hidden_channels] -> [batch_size, hidden_channels]
            h2 = global_mean_pool(h2, batch)
            # [batch_size * nodes, hidden_channels] -> [batch_size, nodes, hidden_channels]
            # h2 = h2.view(max(batch)+1, self.config["num_nbrs"]+1, -1)
            # h2 = h2.view(h2.shape[0], -1)
            # h2 = F.tanh(self.norm3(self.lin(h2)))

        out = self.readout(h2)
        return out


class DenseGCN(nn.Module):
    def __init__(self, config):
        super(DenseGCN, self).__init__()
        self.config = config
        self.conv1 = DenseGCNConv(self.config["input_dim"], 128)
        self.norm1 = LayerNorm(128)
        self.conv2 = DenseGCNConv(128, 128)
        self.norm2 = LayerNorm(128)
        self.lin = nn.Linear(128*(self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(# nn.Linear(len(self.config["multiview"])*64*(self.config["num_nbrs"]+1), 32),
                                     nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, A, H):
        H_list = []
        for i in range(A.shape[-1]):
            Ai, Hi = A[:, :, :, i], H[:, :, :, i]
            Hi = F.relu(self.norm1(self.conv1(Hi, Ai)))
            Hi = F.relu(self.norm2(self.conv2(Hi, Ai)))
            Hi = Hi.view(Hi.shape[0], -1)
            Hi = F.relu(self.lin(Hi))
            H_list.append(Hi)

        # z = torch.cat(H_list, dim=-1)
        z = torch.mean(torch.stack(H_list), dim=0)
        out = self.readout(z)
        return out


# GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, config):
        super(GraphSAGE, self).__init__()
        self.config = config
        self.conv1 = SAGEConv(self.config["input_dim"], 128)
        self.norm1 = LayerNorm(128)
        self.conv2 = SAGEConv(128, 128)
        self.norm2 = LayerNorm(128)
        self.lin = nn.Linear(128 * (self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight, batch=None):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))
        if batch is not None:
            # [batch_size * nodes, hidden_channels] -> [batch_size, hidden_channels]
            h2 = global_mean_pool(h2, batch)
            # [batch_size * nodes, hidden_channels] -> [batch_size, nodes, hidden_channels]
            # h2 = h2.view(max(batch)+1, self.config["num_nbrs"]+1, -1)
            # h2 = h2.view(h2.shape[0], -1)
            # h2 = F.tanh(self.norm3(self.lin(h2)))
        out = self.readout(h2)
        return out


class DenseGraphSAGE(nn.Module):
    def __init__(self, config):
        super(DenseGraphSAGE, self).__init__()
        self.config = config
        self.conv1 = DenseSAGEConv(self.config["input_dim"], 128)
        self.norm1 = LayerNorm(128)
        self.conv2 = DenseSAGEConv(128, 128)
        self.norm2 = LayerNorm(128)
        self.lin = nn.Linear(128*(self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(# nn.Linear(len(self.config["multiview"])*64*(self.config["num_nbrs"]+1), 32),
                                     nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, A, H):
        H_list = []
        for i in range(A.shape[-1]):
            Ai, Hi = A[:, :, :, i], H[:, :, :, i]
            Hi = F.relu(self.norm1(self.conv1(Hi, Ai)))
            Hi = F.relu(self.norm2(self.conv2(Hi, Ai)))
            Hi = Hi.view(Hi.shape[0], -1)
            Hi = F.relu(self.norm3(self.lin(Hi)))
            H_list.append(Hi)

        # z = torch.cat(H_list, dim=-1)
        z = torch.mean(torch.stack(H_list), dim=0)
        out = self.readout(z)
        return out


# GAT
class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        self.config = config
        self.heads = 4
        self.conv1 = GATConv(self.config["input_dim"], 128, heads=self.heads)
        self.norm1 = LayerNorm(128*self.heads)
        self.conv2 = GATConv(128*self.heads, 128)
        self.norm2 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight, batch=None):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))     #################
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))    #################
        if batch is not None:
            h2 = global_mean_pool(h2, batch)
        out = self.readout(h2)
        return out


# SuperGAT
class SuperGAT(nn.Module):
    def __init__(self, config):
        super(SuperGAT, self).__init__()
        self.config = config
        self.heads = 4
        self.conv1 = SuperGATConv(self.config["input_dim"], 128, heads=self.heads)
        self.norm1 = LayerNorm(128 * self.heads)
        self.conv2 = SuperGATConv(128*self.heads, 128)
        self.norm2 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))
        out = self.readout(h2)
        return out


# GIN
class GIN(nn.Module):
    def __init__(self, config):
        super(GIN, self).__init__()
        self.config = config

        self.mlp1 = nn.Sequential(nn.Linear(self.config["input_dim"], 128),
                                  nn.LayerNorm(128),
                                  nn.ReLU(),
                                  nn.Linear(128, 128))
        self.conv1 = GINConv(self.mlp1, train_eps=True)
        self.norm1 = LayerNorm(128)

        self.mlp2 = nn.Sequential(nn.Linear(128, 128),
                                  nn.LayerNorm(128),  # BatchNorm1d
                                  nn.ReLU(),
                                  nn.Linear(128, 128))
        self.conv2 = GINConv(self.mlp2, train_eps=True)
        self.norm2 = LayerNorm(128)

        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))
        out = self.readout(h2)
        return out


class DenseGIN(nn.Module):
    def __init__(self, config):
        super(DenseGIN, self).__init__()
        self.config = config

        self.mlp1 = nn.Sequential(nn.Linear(self.config["input_dim"], 128),
                                  nn.LayerNorm(128),
                                  nn.ReLU(),
                                  nn.Linear(128, 128))
        self.conv1 = DenseGINConv(self.mlp1, train_eps=True)
        self.norm1 = LayerNorm(128)

        self.mlp2 = nn.Sequential(nn.Linear(128, 128),
                                  nn.LayerNorm(128),  # BatchNorm1d
                                  nn.ReLU(),
                                  nn.Linear(128, 128))
        self.conv2 = DenseGINConv(self.mlp2, train_eps=True)
        self.norm2 = LayerNorm(128)

        self.lin = nn.Linear(128*(self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(# nn.Linear(len(self.config["multiview"])*64*(self.config["num_nbrs"]+1), 32),
                                     nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, A, H):
        H_list = []
        for i in range(A.shape[-1]):
            Ai, Hi = A[:, :, :, i], H[:, :, :, i]
            Hi = F.relu(self.norm1(self.conv1(Hi, Ai)))
            Hi = F.relu(self.norm2(self.conv2(Hi, Ai)))
            Hi = Hi.view(Hi.shape[0], -1)
            Hi = F.relu(self.norm3(self.lin(Hi)))
            H_list.append(Hi)

        # z = torch.cat(H_list, dim=-1)
        z = torch.mean(torch.stack(H_list), dim=0)
        out = self.readout(z)
        return out


# GCN-II
class GCNII(nn.Module):
    def __init__(self, config):
        super(GCNII, self).__init__()
        self.config = config
        self.linear = nn.Linear(self.config["input_dim"], 128)
        self.conv1 = GCN2Conv(128, 0.5)
        self.norm1 = LayerNorm(128)
        self.conv2 = GCN2Conv(128, 0.5)
        self.norm2 = LayerNorm(128)
        self.lin = nn.Linear(128 * (self.config["num_nbrs"] + 1), 128)
        self.norm3 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight, batch=None):
        x = x0 = F.relu(self.linear(x))
        h1 = F.relu(self.norm1(self.conv1(x, x0, edge_index)))
        h2 = F.relu(self.norm2(self.conv2(h1, x0, edge_index)))
        if batch is not None:
            # [batch_size * nodes, hidden_channels] -> [batch_size, hidden_channels]
            h2 = global_mean_pool(h2, batch)
            # [batch_size * nodes, hidden_channels] -> [batch_size, nodes, hidden_channels]
            # h2 = h2.view(max(batch) + 1, self.config["num_nbrs"] + 1, -1)
            # h2 = h2.view(h2.shape[0], -1)
            # h2 = F.tanh(self.norm3(self.lin(h2)))
        out = self.readout(h2)
        return out


# EGC
class EGC(nn.Module):
    def __init__(self, config):
        super(EGC, self).__init__()
        self.config = config
        self.conv1 = EGConv(self.config["input_dim"], 128, aggregators=["mean", "symnorm", "var", "std", "max"])
        self.norm1 = LayerNorm(128)
        self.conv2 = EGConv(128, 128, aggregators=["mean", "symnorm", "var", "std", "max"])
        self.norm2 = LayerNorm(128)
        self.readout = nn.Sequential(nn.Linear(128, 32),
                                     nn.LayerNorm(32),
                                     nn.ReLU(),
                                     nn.Linear(32, self.config["output_dim"]))

    def forward(self, x, edge_index, edge_weight):
        h1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        h2 = F.relu(self.norm2(self.conv2(h1, edge_index)))
        out = self.readout(h2)
        return out

