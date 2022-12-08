import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, DenseGINConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SuperGATConv, GINConv, GCN2Conv, EGConv


# GCN
class GCN5(nn.Module):
    def __init__(self, config):
        super(GCN5, self).__init__()
        self.config = config

        self.lg_conv1 = DenseGCNConv(self.config["input_dim"], 128)
        self.lg_norm1 = LayerNorm(128)
        self.lg_conv2 = DenseGCNConv(128, 128)
        self.lg_norm2 = LayerNorm(128)
        self.lin = nn.Linear(128*(self.config["num_nbrs"] + 1), 64)
        self.lin_norm = LayerNorm(64)

        self.conv1 = GCNConv(64, 128)
        self.gg_norm1 = LayerNorm(128)
        self.conv2 = GCNConv(128, 64)
        self.gg_norm2 = LayerNorm(64)
        self.readout = nn.Sequential(nn.Linear(128, self.config["output_dim"]))

    def forward(self, batch_A, batch_H, edge_index, edge_weight):
        H = F.relu(self.lg_norm1(self.lg_conv1(batch_H, batch_A)))
        H = F.relu(self.lg_norm2(self.lg_conv2(H, batch_A)))
        H = H.view(H.shape[0], -1)
        z = F.selu(self.lin_norm(self.lin(H)))

        z2 = F.relu(self.gg_norm1(self.conv1(z, edge_index)))
        z2 = F.relu(self.gg_norm2(self.conv2(z2, edge_index)))
        all_emb = torch.cat((z, z2), dim=1)
        out = self.readout(all_emb)
        return out


# GraphSAGE
class GraphSAGE5(nn.Module):
    def __init__(self, config):
        super(GraphSAGE5, self).__init__()
        self.config = config

        self.lg_conv1 = DenseSAGEConv(self.config["input_dim"], 128)
        self.lg_norm1 = LayerNorm(128)
        self.lg_conv2 = DenseSAGEConv(128, 128)
        self.lg_norm2 = LayerNorm(128)
        self.lin = nn.Linear(128*(self.config["num_nbrs"] + 1), 64)
        self.lin_norm = LayerNorm(64)

        self.conv1 = SAGEConv(64, 128)
        self.gg_norm1 = LayerNorm(128)
        self.conv2 = SAGEConv(128, 64)
        self.gg_norm2 = LayerNorm(64)
        self.readout = nn.Sequential(nn.Linear(128, self.config["output_dim"]))

    def forward(self, batch_A, batch_H, edge_index, edge_weight):
        H = F.relu(self.lg_norm1(self.lg_conv1(batch_H, batch_A)))
        H = F.relu(self.lg_norm2(self.lg_conv2(H, batch_A)))
        H = H.view(H.shape[0], -1)
        z = F.selu(self.lin_norm(self.lin(H)))

        z2 = F.relu(self.gg_norm1(self.conv1(z, edge_index)))
        z2 = F.relu(self.gg_norm2(self.conv2(z2, edge_index)))
        all_emb = torch.cat((z, z2), dim=1)
        out = self.readout(all_emb)
        return out
