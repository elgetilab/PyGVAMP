"""Microbenchmark for the Tier 1 graph-build vectorization.

Runs the new vectorized `_create_graph_from_frame` and the frozen
legacy implementation side-by-side on representative protein shapes,
then times a full `__getitem__` pass through a synthetic dataset to
verify the cached node features + pre-converted frames also pay off.

Run:
    python local_checks/bench_graph_build.py

No GPU required. No real trajectory data required. Output is printed
to stdout; nothing is saved.
"""
from __future__ import annotations

import time

import torch
from torch_geometric.data import Data

from pygv.dataset.vampnet_dataset import VAMPNetDataset


# ---------------------------------------------------------------------------
# Frozen legacy implementation (copied verbatim from
# tests/test_dataset.py::_legacy_build_graph — kept inline so this script
# is fully self-contained and doesn't depend on the test module).
# ---------------------------------------------------------------------------
def _legacy_build_graph(coords, n_neighbors, n_atoms, distance_min,
                        distance_max, gaussian_expansion_dim, gaussian_var,
                        node_attr):
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distances = torch.sqrt((diff ** 2).sum(dim=2))
    diag_mask = torch.eye(n_atoms, dtype=torch.bool)
    distances[diag_mask] = -1.0
    valid_mask = ~diag_mask

    nn_indices = []
    for i in range(n_atoms):
        valid_distances = distances[i][valid_mask[i]]
        valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
        _, top_k = torch.topk(
            valid_distances,
            min(n_neighbors, len(valid_distances)),
            largest=False,
        )
        nn_indices.append(valid_indices[top_k])
    nn_indices = torch.stack(nn_indices)

    edge_set = set()
    for i in range(n_atoms):
        for j in nn_indices[i]:
            edge_set.add((i, j.item()))
    directional = [(t, s) for s, t in edge_set]
    src = torch.tensor([e[0] for e in directional])
    tgt = torch.tensor([e[1] for e in directional])
    edge_index = torch.stack([src, tgt], dim=0)
    edge_dist = distances[src, tgt]

    K = gaussian_expansion_dim
    sigma = (
        gaussian_var
        if gaussian_var is not None
        else (distance_max - distance_min) / K
    )
    mu = torch.linspace(distance_min, distance_max, K)
    edge_attr = torch.exp(-((edge_dist.unsqueeze(-1) - mu) ** 2) / sigma ** 2)

    return Data(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_atoms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_stub_single_frame(n_atoms, n_neighbors, coords):
    """Minimal VAMPNetDataset shell holding a single frame.

    Bypasses __init__ — sets only the attributes that
    _create_graph_from_frame and its helpers read.
    """
    ds = VAMPNetDataset.__new__(VAMPNetDataset)
    ds.n_atoms = n_atoms
    ds.n_neighbors = n_neighbors
    ds.distance_min = 0.0
    ds.distance_max = 3.0
    ds.gaussian_expansion_dim = 16
    ds.gaussian_var = None
    ds.use_amino_acid_encoding = False
    ds.amino_acid_feature_type = "labels"
    ds._frames_t = coords.unsqueeze(0)
    ds._node_attr_cache = {False: torch.eye(n_atoms)}
    ds._rbf_centers = None
    ds._rbf_inv_sigma2 = None
    return ds


def _make_stub_many_frames(n_atoms, n_neighbors, n_frames, seed=0):
    """Dataset shell with many frames and t0/t1 indices set, supports __getitem__."""
    torch.manual_seed(seed)
    ds = _make_stub_single_frame(n_atoms, n_neighbors, torch.zeros(n_atoms, 3))
    ds._frames_t = torch.rand(n_frames, n_atoms, 3) * 3.0
    ds.t0_indices = list(range(n_frames - 1))
    ds.t1_indices = list(range(1, n_frames))
    return ds


def _time(fn, n_iters):
    """Return per-call seconds, with warmup."""
    for _ in range(min(20, max(1, n_iters // 10))):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters


# ---------------------------------------------------------------------------
# Benchmark 1: single graph build, new vs legacy
# ---------------------------------------------------------------------------
def bench_single_build(n_iters=500):
    shapes = [
        ("Trp-cage", 20, 7),
        ("Villin", 35, 10),
        ("Aβ42-ish", 42, 8),
        ("Large", 80, 12),
    ]

    print("=" * 72)
    print("Benchmark 1: per-call _create_graph_from_frame timing")
    print(f"({n_iters} iterations after warmup, lower is better)")
    print("=" * 72)
    print(f"{'Shape':<12} {'n_atoms':>8} {'k':>4} {'legacy (μs)':>14} {'new (μs)':>11} {'speedup':>9}")
    print("-" * 72)

    for name, n_atoms, k in shapes:
        torch.manual_seed(0)
        coords = torch.rand(n_atoms, 3) * 3.0
        node_attr = torch.eye(n_atoms)

        ds = _make_stub_single_frame(n_atoms, k, coords)
        # Force RBF buffers to be built so they don't pollute the first timing call.
        ds._create_graph_from_frame(0)

        legacy_t = _time(
            lambda: _legacy_build_graph(
                coords.clone(), k, n_atoms,
                ds.distance_min, ds.distance_max,
                ds.gaussian_expansion_dim, ds.gaussian_var,
                node_attr,
            ),
            n_iters,
        )
        new_t = _time(lambda: ds._create_graph_from_frame(0), n_iters)

        speedup = legacy_t / new_t if new_t > 0 else float('inf')
        print(f"{name:<12} {n_atoms:>8} {k:>4} {legacy_t*1e6:>14.1f} {new_t*1e6:>11.1f} {speedup:>8.1f}x")


# ---------------------------------------------------------------------------
# Frozen legacy __getitem__ equivalent.
#
# Mirrors what the old VAMPNetDataset.__getitem__ did per call:
#   1. numpy -> torch coord conversion (was repeated every call).
#   2. legacy graph build for t0.
#   3. node feature construction via a Python for-loop (was not cached).
#   4. Same again for t1.
#   5. Return tuple.
#
# Kept inline so the script is self-contained and the comparison is fair —
# the new __getitem__ benefits from cached node features + pre-converted
# frames + vectorized kNN, and this function lets us see the compound win.
# ---------------------------------------------------------------------------
def _legacy_node_features(n_atoms):
    """Replicates the legacy one-hot construction: Python for-loop over diagonal."""
    node_attr = torch.zeros(n_atoms, n_atoms)
    for i in range(n_atoms):
        node_attr[i, i] = 1.0
    return node_attr


def _legacy_getitem(frames_np, t0_indices, t1_indices, n_neighbors, n_atoms,
                    distance_min, distance_max, gaussian_expansion_dim,
                    gaussian_var, idx):
    """End-to-end legacy __getitem__: rebuilds everything per call."""
    t0_idx = t0_indices[idx]
    t1_idx = t1_indices[idx]

    # Per-call numpy -> torch (legacy did this on every access)
    coords_t0 = torch.tensor(frames_np[t0_idx], dtype=torch.float32)
    coords_t1 = torch.tensor(frames_np[t1_idx], dtype=torch.float32)
    # Per-call node-feature build (legacy did NOT cache)
    node_attr_t0 = _legacy_node_features(n_atoms)
    node_attr_t1 = _legacy_node_features(n_atoms)

    g0 = _legacy_build_graph(
        coords_t0, n_neighbors, n_atoms, distance_min, distance_max,
        gaussian_expansion_dim, gaussian_var, node_attr_t0,
    )
    g1 = _legacy_build_graph(
        coords_t1, n_neighbors, n_atoms, distance_min, distance_max,
        gaussian_expansion_dim, gaussian_var, node_attr_t1,
    )
    return g0, g1


# ---------------------------------------------------------------------------
# Benchmark 2: end-to-end __getitem__, new vs legacy
# ---------------------------------------------------------------------------
def bench_getitem(n_iters=500):
    print()
    print("=" * 72)
    print("Benchmark 2: end-to-end __getitem__ (new vs legacy)")
    print(f"({n_iters} calls, Trp-cage shape: n_atoms=20, k=7)")
    print("=" * 72)

    n_frames = 1000
    n_atoms, k = 20, 7

    ds = _make_stub_many_frames(n_atoms=n_atoms, n_neighbors=k, n_frames=n_frames)
    # Numpy view of the same frames for the legacy path.  Sharing data
    # underneath keeps the comparison apples-to-apples.
    frames_np = ds._frames_t.numpy()

    # Warmup both paths so RBF buffers are built and any first-call costs
    # (Python tier-up, kernel JIT) are amortized.
    for i in range(20):
        ds[i]
        _legacy_getitem(
            frames_np, ds.t0_indices, ds.t1_indices, k, n_atoms,
            ds.distance_min, ds.distance_max,
            ds.gaussian_expansion_dim, ds.gaussian_var, i,
        )

    # New path
    t0 = time.perf_counter()
    for i in range(n_iters):
        ds[i]
    new_elapsed = time.perf_counter() - t0
    new_per = new_elapsed / n_iters

    # Legacy path
    t0 = time.perf_counter()
    for i in range(n_iters):
        _legacy_getitem(
            frames_np, ds.t0_indices, ds.t1_indices, k, n_atoms,
            ds.distance_min, ds.distance_max,
            ds.gaussian_expansion_dim, ds.gaussian_var, i,
        )
    legacy_elapsed = time.perf_counter() - t0
    legacy_per = legacy_elapsed / n_iters

    speedup = legacy_per / new_per if new_per > 0 else float('inf')

    print(f"{'metric':<32} {'legacy':>12} {'new':>12} {'speedup':>10}")
    print("-" * 72)
    print(f"{'per __getitem__ (μs)':<32} {legacy_per*1e6:>12.1f} "
          f"{new_per*1e6:>12.1f} {speedup:>9.1f}x")
    print(f"{'throughput (samples/sec)':<32} {1/legacy_per:>12.0f} "
          f"{1/new_per:>12.0f}")
    print(f"{'implied batch=1000 (ms)':<32} {1000*legacy_per*1e3:>12.1f} "
          f"{1000*new_per*1e3:>12.1f}")
    print()
    print("Notes:")
    print("- batch=1000 build × 730 batches/epoch ≈ single-worker per-epoch")
    print("  graph-build cost.  With DataLoader workers running in parallel,")
    print("  real-world time scales near 1/n_workers down to GPU/IO limits.")
    print("- The new path's win over legacy compounds three changes: vectorized")
    print("  kNN, cached frame-invariant node features, and pre-converted")
    print("  frames tensor.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bench_single_build()
    bench_getitem()
