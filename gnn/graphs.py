"""Sensor graphs. Nodes are the 14 sensors; four edge sets are compared.

* physics:     sensors linked when they sit on the same component, on adjacent stages of the
               gas path, or on the same shaft
* correlation: each sensor linked to its 3 most correlated sensors on training engines
* full:        every pair of sensors linked (attention has to find the structure itself)
* none:        self-loops only, so no information passes between sensors (ablation)
"""
from itertools import combinations

import numpy as np
import torch

from .data import NODE_NAMES

# Gas path: fan -> LPC -> HPC -> combustor -> HPT -> LPT
COMPONENTS = {
    "fan": ["Nf", "NRf", "BPR"],
    "lpc": ["T24"],
    "hpc": ["T30", "P30", "Ps30", "htBleed"],
    "combustor": ["phi"],
    "hpt": ["W31"],
    "lpt": ["T50", "W32"],
}
GAS_PATH = ["fan", "lpc", "hpc", "combustor", "hpt", "lpt"]
SHAFTS = {
    "low_pressure": ["Nf", "NRf", "T24", "T50", "W32"],   # fan + LPC driven by the LPT
    "high_pressure": ["Nc", "NRc", "T30", "P30", "W31"],  # HPC driven by the HPT
}
BLEEDS = [("htBleed", "W31"), ("htBleed", "W32")]

INDEX = {name: i for i, name in enumerate(NODE_NAMES)}


def _to_edge_index(pairs):
    """Undirected pairs -> PyG edge_index with both directions and no duplicates."""
    pairs = list(pairs)
    edges = sorted({(INDEX[a], INDEX[b]) for a, b in pairs if a != b} |
                   {(INDEX[b], INDEX[a]) for a, b in pairs if a != b})
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def physics_edges():
    pairs = []
    for members in COMPONENTS.values():
        pairs += combinations(members, 2)
    for up, down in zip(GAS_PATH, GAS_PATH[1:]):
        pairs += [(a, b) for a in COMPONENTS[up] for b in COMPONENTS[down]]
    for members in SHAFTS.values():
        pairs += combinations(members, 2)
    pairs += BLEEDS
    return _to_edge_index(pairs)


def correlation_edges(train_df, k=3):
    corr = np.abs(np.corrcoef(train_df[NODE_NAMES].to_numpy().T))
    np.fill_diagonal(corr, -1)
    pairs = [(NODE_NAMES[i], NODE_NAMES[j]) for i in range(len(NODE_NAMES)) for j in np.argsort(corr[i])[-k:]]
    return _to_edge_index(pairs)


def full_edges():
    return _to_edge_index(combinations(NODE_NAMES, 2))


def no_edges():
    return torch.empty((2, 0), dtype=torch.long)


def build_graphs(train_df):
    return {"none": no_edges(), "correlation": correlation_edges(train_df),
            "physics": physics_edges(), "full": full_edges()}
