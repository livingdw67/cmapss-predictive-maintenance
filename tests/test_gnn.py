"""Checks for the GNN data and graph code that need no downloaded data."""
import numpy as np
import pandas as pd
import torch

from gnn.data import NODE_NAMES, make_windows
from gnn.graphs import build_graphs, full_edges, no_edges, physics_edges
from gnn.model import SensorGAT


def _engines(lengths):
    rows = []
    for e, n in enumerate(lengths):
        for c in range(1, n + 1):
            rows.append({"ENGINE_ID": f"E{e}", "CYCLE": c, **{s: 100 * e + c for s in NODE_NAMES}})
    return pd.DataFrame(rows)


def test_windows_end_at_their_row_and_never_cross_engines():
    df = _engines([5, 40])
    X = make_windows(df, window=10)
    assert X.shape == (45, len(NODE_NAMES), 10)
    # Row 2 of engine 0: padded with cycle 1, then cycles 1..3
    assert X[2, 0].tolist() == [1] * 7 + [1, 2, 3]
    # First row of engine 1 must not contain engine 0's readings
    assert set(X[5, 0].tolist()) == {101}
    # Last row of engine 1 is cycles 31..40
    assert X[-1, 0].tolist() == [100 + c for c in range(31, 41)]


def test_graphs_are_symmetric_without_self_loops():
    for edges in [physics_edges(), full_edges()]:
        pairs = set(map(tuple, edges.t().tolist()))
        assert all((d, s) in pairs for s, d in pairs)
        assert all(s != d for s, d in pairs)
    n = len(NODE_NAMES)
    assert full_edges().size(1) == n * (n - 1)
    assert no_edges().size(1) == 0


def test_physics_graph_connects_every_sensor():
    edges = physics_edges()
    assert set(edges[0].tolist()) == set(range(len(NODE_NAMES)))


def test_correlation_graph_built_from_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, len(NODE_NAMES))), columns=NODE_NAMES)
    assert build_graphs(df)["correlation"].size(1) >= 2 * len(NODE_NAMES)


def test_samples_in_a_batch_do_not_leak_into_each_other():
    torch.manual_seed(0)
    model = SensorGAT(len(NODE_NAMES), physics_edges()).eval()
    x = torch.randn(3, len(NODE_NAMES), 30)
    with torch.no_grad():
        together = model(x)[0]
        alone = torch.cat([model(x[i:i + 1])[0] for i in range(3)])
    assert torch.allclose(together, alone, atol=1e-5)
