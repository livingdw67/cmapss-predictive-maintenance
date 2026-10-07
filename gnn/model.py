"""Spatio-temporal GNN: a GRU reads each sensor's history, GAT layers pass messages between sensors."""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class SensorGAT(nn.Module):
    """
    Input:  x of shape (batch, sensors, window), one graph per sample, same edges for every graph.
    Output: (rul, early_warning_logit, critical_action_logit) per graph. RUL is scaled to [0, 1].
    """

    def __init__(self, n_sensors, edge_index, hidden=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.register_buffer("edge_index", edge_index)
        self.sensor_embedding = nn.Embedding(n_sensors, 8)  # lets shared weights tell sensors apart
        self.temporal = nn.GRU(1, 32, batch_first=True)
        self.project = nn.Linear(32 + 8, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=True)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def _batched_edges(self, batch_size):
        # One copy of the sensor graph per sample, offset so the copies don't touch
        offsets = torch.arange(batch_size, device=self.edge_index.device).repeat_interleave(self.edge_index.size(1))
        return self.edge_index.repeat(1, batch_size) + offsets * self.n_sensors

    def forward(self, x, return_attention=False):
        batch_size, n_sensors, window = x.shape
        _, h = self.temporal(x.reshape(-1, window, 1))
        ids = torch.arange(n_sensors, device=x.device).repeat(batch_size)
        h = self.project(torch.cat([h[-1], self.sensor_embedding(ids)], dim=1))

        edge_index = self._batched_edges(batch_size)
        attention = []
        for conv, norm in zip(self.convs, self.norms):
            out, (att_edges, alpha) = conv(h, edge_index, return_attention_weights=True)
            attention.append((att_edges, alpha))
            h = norm(h + nn.functional.elu(out))  # residual

        graph = torch.arange(batch_size, device=x.device).repeat_interleave(n_sensors)
        pooled = torch.cat([global_mean_pool(h, graph), global_max_pool(h, graph)], dim=1)
        out = self.head(pooled)
        rul, ew, ca = torch.sigmoid(out[:, 0]), out[:, 1], out[:, 2]
        return (rul, ew, ca, attention) if return_attention else (rul, ew, ca)
