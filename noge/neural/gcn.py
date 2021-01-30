import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCN_Model(nn.Module):
    def __init__(self, dims, activate_last=False, normalize=False, max_edges=10000, device='cpu', improved=False):
        super().__init__()
        assert len(dims) > 2

        self.normalize = normalize
        self.improved = improved
        self.input_layer = GCNConv(dims[0], dims[1], normalize=False)

        dims = dims[1:]
        hidden_layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            hidden_layers.append(GCNConv(d1, d2, normalize=False))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.activate_last = activate_last

        self._edge_norm = torch.ones((max_edges,), dtype=torch.float32, device=device)

    def forward(self, x, edge_index):

        num_edges = edge_index.size(1)
        edge_weight = self._edge_norm[:num_edges]
        edge_index, edge_weight = gcn_norm(edge_index=edge_index,
                                           num_nodes=x.size(0),
                                           edge_weight=edge_weight,
                                           dtype=x.dtype,
                                           improved=self.improved)

        x = self.input_layer(x, edge_index, edge_weight)
        for layer in self.hidden_layers:
            x = F.relu(x)
            x = layer(x, edge_index, edge_weight=edge_weight)

        if self.activate_last:
            x = F.relu(x)

        return x
