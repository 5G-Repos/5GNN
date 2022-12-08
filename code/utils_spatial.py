
import torch
import numpy as np
import torch.nn.parallel


def latlon_to_cart(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    cart_coord = np.column_stack((x, y, z))
    return cart_coord


def get_Distance(a, b, method="haversine"):
    """
    Calculate the distance between two data points effectively based on the given method
    :param
        a: tensor of "from" node. The dim is [#samples, d]
        b: tensor of "to" node. The dim is [#samples, d]
    :return
        distance: tensor of distance between "from" node and "to" node
    """

    if method == "euclidean":
        distance = torch.sqrt(torch.sum((a-b)**2, dim=1))
    elif method == "haversine":
        loc1, loc2 = torch.deg2rad(a), torch.deg2rad(b)
        dloc = loc1 - loc2
        dlon, dlat = dloc[:, 0], dloc[:, 1]

        a = torch.sin(dlat/2)**2 + torch.cos(loc1[:, 1])*torch.cos(loc2[:, 1])*torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        r = 6371
        distance = c * r * 1000
    return distance


def makeEdgeWeight(c, edge_index):
    """
    Description goes here
    :param
        c:
        edge_index:
    :return
        edge_weight:
    """

    # Get edge weight
    to = edge_index[0]
    fro = edge_index[1]
    edge_weight = get_Distance(c[fro], c[to])
    edge_weight = edge_weight.tolist()

    # Normalize the weight
    max_val = max(edge_weight)
    rng = max_val - min(edge_weight)
    edge_weight = [(max_val-elem)/rng for elem in edge_weight]
    return torch.Tensor(edge_weight)


def knn_to_adj(edge_index, n):
    """
    knn graph to adjacency matrix
    :param
        edge_index:
        n:
    :return:
    """
    adj_matrix = torch.zeros(n, n)  # lil_matrix((n, n), dtype=float)
    for i in range(len(edge_index[0])):
        tow = edge_index[0][i]
        fro = edge_index[1][i]
        adj_matrix[tow, fro] = 1  # should be bidectional?
    return adj_matrix.T


def normal_torch(tensor, min_val=0):
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    if t_min == 0 and t_max == 0:
        return torch.tensor(tensor)
    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    if min_val == 0:
        tensor_norm = ((tensor - t_min) / (t_max - t_min))
    return torch.tensor(tensor_norm)
