# Graph-build speedup — Tier 1 + Tier 2 implementation plan

**Companion to:** `claude/PYG_GRAPH_PREBUILD_INVESTIGATION.md`
**Scope:** dataset hot-path vectorization + DataLoader configuration. No
cache format change, no model change, no CLI change.

**Acceptance bar:** a single Trp-cage v1 seed at 2 CPUs/job runs at
≤ 5 min/epoch (down from ~15 min). Existing test suite passes. New
regression test pins exact mathematical equivalence to the pre-change
behavior (modulo edge ordering — see "Why the `set` is a bug, not a
feature" below).

---

## Why the `set` is a bug, not a feature

Current code, `pygv/dataset/vampnet_dataset.py:439-449`:

```python
edge_set = set()
for i in range(self.n_atoms):
    for j in nn_indices[i]:
        edge_set.add((i, j.item()))

directional_edges = [(target, source) for source, target in edge_set]
source_indices = torch.tensor([edge[0] for edge in directional_edges], ...)
target_indices = torch.tensor([edge[1] for edge in directional_edges], ...)
edge_index = torch.stack([source_indices, target_indices], dim=0)
```

The `set` looks like deduplication, but:

1. `i` is unique per outer-loop iteration (`range(self.n_atoms)`).
2. `nn_indices[i]` comes from `torch.topk`, which returns distinct
   indices per row.
3. So every `(i, j)` pair inserted is already unique. **The set
   deduplicates nothing.**

What the `set` *does* do, however, is two harmful things:

- **Forces a CPU sync per edge** via `j.item()`. With 20 atoms × 7
  neighbors = 140 edges per graph, that's 140 syncs per graph. The PyG
  Data builder never needs Python scalars; it wants a tensor.
- **Makes edge ordering process-dependent.** CPython's `set` iteration
  order depends on the hash of its elements. For `(int, int)` tuples,
  the hash is stable within a single process but **not across processes**
  (the hash seed is randomized by default; only fixed if you set
  `PYTHONHASHSEED`). So today, the same dataset + same seed produces
  different `edge_index` *orderings* in different worker processes —
  the math is identical, but raw tensor equality fails.

The vectorized replacement is the obvious one:

```python
distances.fill_diagonal_(float('inf'))
_, nn_indices = torch.topk(distances, self.n_neighbors, dim=1, largest=False)
# nn_indices: (n_atoms, k)

src = torch.arange(self.n_atoms, device=distances.device) \
        .unsqueeze(1).expand(-1, self.n_neighbors).reshape(-1)
tgt = nn_indices.reshape(-1)
# After the existing source/target swap for message-passing direction:
edge_index = torch.stack([tgt, src], dim=0)
```

This is deterministic, row-major (atom 0's neighbors first, then atom 1,
…), with neighbors within each row in ascending-distance order.

### Sorted-edge equivalence

When we compare "old graph" vs "new graph" for the regression test, we
cannot use `torch.equal(edge_index_old, edge_index_new)` because the old
code's edge ordering is process-dependent and the new code's is
deterministic — they'll usually disagree on column order even when the
underlying graph is identical.

What we *can* assert is:

1. The **set of (src, tgt) pairs** is the same.
2. For every shared (src, tgt) pair, the corresponding `edge_attr` row
   is the same.

Both reduce to: sort the columns of `edge_index` by `(src, tgt)`, apply
the same permutation to `edge_attr`, then compare. Pseudocode:

```python
def canonicalize(graph):
    src, tgt = graph.edge_index[0], graph.edge_index[1]
    key = src * graph.num_nodes + tgt        # unique per pair
    perm = torch.argsort(key)
    return graph.edge_index[:, perm], graph.edge_attr[perm]

old_ei, old_ea = canonicalize(old_graph)
new_ei, new_ea = canonicalize(new_graph)
assert torch.equal(old_ei, new_ei)
assert torch.allclose(old_ea, new_ea, atol=1e-6)
assert torch.allclose(old_graph.x, new_graph.x)
```

This is what "sorted-edge equivalence" means in the test plan below: it
asserts the *graphs are the same*, ignoring the now-fixed ordering
nondeterminism.

GNN math is permutation-equivariant over edges (scatter/aggregate is
order-independent), so this is the correct equivalence to pin for
**model behavior**, not the stricter raw-tensor equality.

---

## Tier 1 — Dataset hot-path vectorization

### Change 1.1: vectorize kNN, drop the `set`

**File:** `pygv/dataset/vampnet_dataset.py:395-469` (`_create_graph_from_frame`)

**Why:** the Python loop + `.item()` syncs + dead-deduplication `set` are
the dominant CPU cost per `__getitem__` call.

**Diff sketch** (logical, not literal):

```diff
 def _create_graph_from_frame(self, frame_idx, use_amino_acid_encoding=None):
     if use_amino_acid_encoding is None:
         use_amino_acid_encoding = self.use_amino_acid_encoding

-    coords = torch.tensor(self.frames[frame_idx], dtype=torch.float32)
+    # self.frames is converted to torch once at load time (see Change 1.3)
+    coords = self._frames_t[frame_idx]

     # Pairwise distances
     diff = coords.unsqueeze(1) - coords.unsqueeze(0)
     distances = torch.sqrt((diff ** 2).sum(dim=2))

-    diag_mask = torch.eye(self.n_atoms, dtype=torch.bool, device=distances.device)
-    distances[diag_mask] = -1.0
-    valid_mask = ~diag_mask
-
-    nn_indices = []
-    for i in range(self.n_atoms):
-        node_distances = distances[i]
-        valid_distances = node_distances[valid_mask[i]]
-        valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
-        _, top_k_indices = torch.topk(
-            valid_distances,
-            min(self.n_neighbors, len(valid_distances)),
-            largest=False,
-        )
-        node_nn_indices = valid_indices[top_k_indices]
-        nn_indices.append(node_nn_indices)
-    nn_indices = torch.stack(nn_indices)
-
-    edge_set = set()
-    for i in range(self.n_atoms):
-        for j in nn_indices[i]:
-            edge_set.add((i, j.item()))
-    directional_edges = [(target, source) for source, target in edge_set]
-    source_indices = torch.tensor([edge[0] for edge in directional_edges], device=distances.device)
-    target_indices = torch.tensor([edge[1] for edge in directional_edges], device=distances.device)
-    edge_index = torch.stack([source_indices, target_indices], dim=0)
+    # Mask self-loops by inflating diagonal distances.  topk(largest=False)
+    # never picks the diagonal because inf > any real distance.  Note:
+    # we do NOT set diagonals to -1.0 here; the gaussian-expansion
+    # function still treats -1 as a sentinel (see Change 1.2) but it
+    # only ever receives off-diagonal entries from the indexing below.
+    distances_self_inf = distances.clone()
+    distances_self_inf.fill_diagonal_(float('inf'))
+
+    k = min(self.n_neighbors, self.n_atoms - 1)
+    _, nn_indices = torch.topk(distances_self_inf, k, dim=1, largest=False)
+    # nn_indices: (n_atoms, k)
+
+    src_row = torch.arange(self.n_atoms, device=distances.device) \
+                   .unsqueeze(1).expand(-1, k).reshape(-1)
+    tgt_col = nn_indices.reshape(-1)
+    # Existing convention: edge_index is [target, source] for
+    # message-passing direction.  Preserve that swap.
+    edge_index = torch.stack([tgt_col, src_row], dim=0)

-    edge_distances = distances[source_indices, target_indices]
+    edge_distances = distances[src_row, tgt_col]
     edge_attr = self._compute_gaussian_expanded_distances(edge_distances)

-    node_attr = self._create_node_features(use_amino_acid_encoding)
+    # Frame-invariant: cached at init time (see Change 1.4)
+    node_attr = self._get_node_features(use_amino_acid_encoding)

     graph = Data(
         x=node_attr,
         edge_index=edge_index,
         edge_attr=edge_attr,
         num_nodes=self.n_atoms,
     )
     return graph
```

**Test impact:**
- `TestGraphConstruction.test_knn_edges_count` — unchanged (count
  identical: `n_atoms × k`).
- `TestGraphConstruction.test_no_self_edges` — unchanged (still
  excludes self).
- `TestGraphConstruction.test_deterministic_graph_construction` —
  **stricter** now, since two builds produce identical edge ordering
  (was only true within a single process before; now true everywhere).
  Still passes.
- New regression test (Change 1.5) pins the cross-version equivalence.

### Change 1.2: vectorize and precompute Gaussian expansion

**File:** `pygv/dataset/vampnet_dataset.py:306-330`

**Why:** `torch.linspace` is recomputed every call; `torch.zeros + gather
+ scatter` is unnecessarily complex; the function is called twice per
`__getitem__` × every batch.

**Constraint:** `test_negative_distance_sentinel_zeroed` pins that
`distance == -1` yields a zero row. The vectorized version must
preserve this.

**Diff sketch:**

```diff
 class VAMPNetDataset(Dataset):
     def __init__(self, ...):
         ...
+        # Precompute RBF parameters as buffers — frame-invariant.
+        # Populated lazily once distance_min/max are known.
+        self._rbf_centers = None
+        self._rbf_inv_sigma2 = None
+
+    def _ensure_rbf_buffers(self, device=None):
+        if self._rbf_centers is not None:
+            return
+        K = self.gaussian_expansion_dim
+        d_range = self.distance_max - self.distance_min
+        sigma = self.gaussian_var if self.gaussian_var is not None else d_range / K
+        self._rbf_centers = torch.linspace(
+            self.distance_min, self.distance_max, K,
+            device=device,
+        )
+        self._rbf_inv_sigma2 = 1.0 / (sigma ** 2)

     def _compute_gaussian_expanded_distances(self, distances):
-        K = self.gaussian_expansion_dim
-        d_range = self.distance_max - self.distance_min
-        sigma = self.gaussian_var if self.gaussian_var is not None else d_range / K
-
-        valid_mask = distances >= 0
-        distances_reshaped = distances.reshape(-1, 1)
-        mu_values = torch.linspace(self.distance_min, self.distance_max, K).view(1, -1)
-
-        expanded_features = torch.zeros(
-            (distances_reshaped.shape[0], K),
-            device=distances.device, dtype=torch.float32,
-        )
-        valid_indices = torch.nonzero(valid_mask).squeeze()
-        valid_distances = distances_reshaped[valid_indices]
-        valid_expanded = torch.exp(-((valid_distances - mu_values) ** 2) / (sigma ** 2))
-        expanded_features[valid_indices] = valid_expanded
-        return expanded_features
+        self._ensure_rbf_buffers(device=distances.device)
+        # (N, 1) - (1, K) -> (N, K), elementwise.
+        delta = distances.unsqueeze(-1) - self._rbf_centers.unsqueeze(0)
+        expanded = torch.exp(-(delta ** 2) * self._rbf_inv_sigma2)
+        # Preserve sentinel: distances < 0 -> zero row.
+        # Multiplying by a (N, 1) float mask is cheaper than gather/scatter.
+        valid = (distances >= 0).to(expanded.dtype).unsqueeze(-1)
+        return expanded * valid
```

Note: `_ensure_rbf_buffers` is needed because `distance_min`/`distance_max`
are not known until `_determine_distance_range` runs *inside* `__init__`,
after the buffers field is first set. Lazy init is the simplest fix.

**Test impact:**
- `test_centers_match_reference_at_canonical_settings` — unchanged
  (still uses `linspace` semantics).
- `test_sigma_differs_by_K_over_K_minus_1` — unchanged.
- `test_expansion_diverges_from_reference_under_canonical_settings` —
  unchanged.
- `test_expansion_known_values` — unchanged.
- `test_negative_distance_sentinel_zeroed` — **must still pass**. The
  mask via multiplication preserves the behavior.
- `test_gaussian_var_override_matches_reference` — unchanged.

### Change 1.3: store frames as a torch tensor once

**File:** `pygv/dataset/vampnet_dataset.py:185-222` (`_process_trajectories`)
and `pygv/dataset/vampnet_dataset.py:555-602` (`_load_from_cache`) and
`pygv/dataset/vampnet_dataset.py:665-725` (`from_cache`)

**Why:** `coords = torch.tensor(self.frames[frame_idx], dtype=torch.float32)`
in the hot path copies a numpy slice on every call. A single up-front
conversion is enough.

**Approach:** add `self._frames_t: torch.Tensor` populated at the same
points where `self.frames` (numpy) is set. Keep `self.frames` (numpy)
because the cache pickle stores it as numpy, and `_determine_distance_range`
uses numpy indexing (`self.frames[idx]`); easier to leave it alone.

**Diff sketch:**

```diff
 def _process_trajectories(self):
     ...
     self.frames = np.array(self.frames)
+    self._frames_t = torch.from_numpy(self.frames).to(torch.float32)
     self.n_frames = len(self.frames)
     ...

 def _load_from_cache(self):
     ...
     self.frames = data['frames']
+    self._frames_t = torch.from_numpy(self.frames).to(torch.float32) \
+                          if isinstance(self.frames, np.ndarray) \
+                          else self.frames.to(torch.float32)
     ...
```

(Same one-liner in `from_cache`.)

**Test impact:** none — `self._frames_t[i]` produces a tensor of the same
shape and dtype as `torch.tensor(self.frames[i], dtype=torch.float32)`.

### Change 1.4: cache frame-invariant node features

**File:** `pygv/dataset/vampnet_dataset.py:354-393` (`_create_node_features`)

**Why:** `_create_node_features` is called twice per `__getitem__` and
its output never depends on `frame_idx`. The one-hot branch is a 20×20
identity matrix; the AA branches iterate `self.atom_indices` and read
residue names from a fixed topology.

**Diff sketch:**

```diff
 class VAMPNetDataset(Dataset):
     def __init__(self, ...):
         ...
+        # Lazy cache for frame-invariant node features. Keyed on
+        # (use_amino_acid_encoding,) since get_frames_dataset_with_encoding
+        # can request the opposite encoding for analysis.
+        self._node_attr_cache = {}

-    def _create_node_features(self, use_amino_acid_encoding=None):
+    def _get_node_features(self, use_amino_acid_encoding=None):
         if use_amino_acid_encoding is None:
             use_amino_acid_encoding = self.use_amino_acid_encoding
+        key = bool(use_amino_acid_encoding), self.amino_acid_feature_type
+        cached = self._node_attr_cache.get(key)
+        if cached is not None:
+            return cached
+        cached = self._build_node_features(use_amino_acid_encoding)
+        self._node_attr_cache[key] = cached
+        return cached

+    def _build_node_features(self, use_amino_acid_encoding):
         # (existing body of _create_node_features, unchanged)
         ...
```

The public name `_create_node_features` is preserved as an alias to
`_build_node_features` for any external callers (none found in the repo,
but worth keeping for safety).

**Test impact:**
- `TestNodeFeatures.*` — unchanged in semantics; tests call
  `dataset.get_graph(...)` which now returns a cached tensor. Tests
  inspect values, not identity, so caching is invisible.
- Caveat: tests must not mutate `graph.x` in-place, because they'd
  mutate the cache. None of them do; they only read.

### Change 1.5: regression test pinning sorted-edge equivalence

**File:** `tests/test_dataset.py` (new test class)

**Why:** lock the "vectorized version produces the same graphs as the
old version" guarantee. Without this, future refactors of
`_create_graph_from_frame` could silently change graphs.

**Approach:** keep a frozen copy of the *old* `_create_graph_from_frame`
inline in the test file, run it on a small fixed-coordinate fixture,
and compare to the new version using sorted-edge canonicalization.

**Sketch:**

```python
import torch
import numpy as np
from torch_geometric.data import Data

from pygv.dataset.vampnet_dataset import VAMPNetDataset


def _legacy_build_graph(coords, n_neighbors, n_atoms, distance_min,
                        distance_max, gaussian_expansion_dim, gaussian_var,
                        node_attr):
    """Frozen pre-vectorization implementation. Do not modify."""
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
    sigma = gaussian_var if gaussian_var is not None else (distance_max - distance_min) / K
    mu = torch.linspace(distance_min, distance_max, K)
    edge_attr = torch.exp(-((edge_dist.unsqueeze(-1) - mu) ** 2) / sigma ** 2)

    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=n_atoms)


def _canonicalize(graph):
    src, tgt = graph.edge_index[0], graph.edge_index[1]
    key = src * graph.num_nodes + tgt
    perm = torch.argsort(key)
    return graph.edge_index[:, perm], graph.edge_attr[perm]


class TestGraphBuildRegression:
    """Pin that vectorized graph build matches legacy build mathematically."""

    @pytest.mark.parametrize("seed,n_atoms,n_neighbors", [
        (0, 20, 7),   # Trp-cage shape
        (1, 35, 10),  # Villin shape
        (2, 42, 8),   # Aβ42-ish
        (3, 5, 3),    # Tiny edge case
    ])
    def test_sorted_edge_equivalence(self, mock_mdtraj, dataset_params,
                                     seed, n_atoms, n_neighbors):
        from pygv.dataset.vampnet_dataset import VAMPNetDataset

        torch.manual_seed(seed)
        coords = torch.randn(n_atoms, 3) * 1.5

        # Build a minimal dataset shell to get matching distance_min/max
        # and gaussian parameters.
        params = dataset_params.copy()
        params['n_neighbors'] = n_neighbors
        ds = VAMPNetDataset(**params)
        # Force distance range to a known span so the RBF is well-defined
        ds.distance_min = 0.0
        ds.distance_max = 3.0
        # Bypass cache invalidation: clear any precomputed RBF buffers.
        ds._rbf_centers = None
        ds._rbf_inv_sigma2 = None
        ds.n_atoms = n_atoms
        ds._frames_t = coords.unsqueeze(0)  # 1 frame
        ds.frames = ds._frames_t.numpy()
        ds._node_attr_cache.clear()
        # node features: use a stable shape independent of n_atoms
        node_attr = torch.eye(n_atoms)
        ds._node_attr_cache[(False, ds.amino_acid_feature_type)] = node_attr

        new_graph = ds._create_graph_from_frame(0)
        old_graph = _legacy_build_graph(
            coords, n_neighbors, n_atoms,
            ds.distance_min, ds.distance_max,
            ds.gaussian_expansion_dim, ds.gaussian_var,
            node_attr,
        )

        new_ei, new_ea = _canonicalize(new_graph)
        old_ei, old_ea = _canonicalize(old_graph)

        assert new_graph.num_nodes == old_graph.num_nodes
        assert new_graph.edge_index.shape == old_graph.edge_index.shape
        assert torch.equal(new_ei, old_ei), (
            f"Edge sets diverge for seed={seed}, n_atoms={n_atoms}"
        )
        assert torch.allclose(new_ea, old_ea, atol=1e-6)
        assert torch.allclose(new_graph.x, old_graph.x)

    def test_vectorized_edge_ordering_is_deterministic(self, mock_mdtraj,
                                                       dataset_params):
        """New code: edges should be in row-major order (atom 0 first, etc.).
        This is the property that makes raw tensor equality stable across
        processes — locked here so we notice if it regresses."""
        from pygv.dataset.vampnet_dataset import VAMPNetDataset
        ds = VAMPNetDataset(**dataset_params)
        g = ds.get_graph(0)
        # edge_index is [target, source] after the message-passing swap,
        # so the *source* row is the row-major iteration variable.
        src_row = g.edge_index[1]
        # Check src_row is non-decreasing — i.e. atom-major order.
        assert torch.all(src_row[1:] >= src_row[:-1]), (
            "Expected source indices to be non-decreasing (row-major build)"
        )
```

**Coverage gap to flag:** this test pins the *math* of one graph build,
not the end-to-end VAMP score. End-to-end is covered indirectly by
running a small training smoke test (existing
`test_training.py`/`test_pipeline_integration.py`), which exercises the
full path with `num_workers=0`. After the Tier 1+2 changes land, those
smoke tests should pass without modification.

### Change 1.6: remove dead `node_embeddings` attribute (optional)

`pygv/dataset/vampnet_dataset.py:153-154, 332-352`: `self.node_embeddings`
is set as a `nn.Parameter` then immediately overwritten to a plain tensor
in `_initialize_node_embeddings()`, and is never read anywhere outside
the module. This is dead code.

Recommendation: **leave it in this PR** to keep the diff focused on perf.
Track in a follow-up cleanup ticket.

---

## Tier 2 — DataLoader configuration

### Change 2.1: bump `num_workers`, kill the test loader's idle worker

**File:** `pygv/pipe/training.py:158-185`

**Why:** at `--cpus-per-task=2`, `available_cpus() // 2 == 1` worker per
loader. Two persistent workers (one for train, one for test) both
occupy CPU slots, but only the train worker runs during the training
loop. The test worker is dead weight during the hot path.

Also: no `prefetch_factor` is specified (defaults to 2). With 1 worker
and prefetch=2, only 2 batches can be queued. Bumping prefetch is free
when workers are CPU-bound producers.

**Diff:**

```diff
-    train_loader = DataLoader(
-        train_dataset,
-        shuffle=True,
-        batch_size=args.batch_size,
-        drop_last=True,
-        pin_memory=torch.cuda.is_available() and not args.cpu,
-        num_workers=max(1, available_cpus() // 2),
-        persistent_workers=True,
-    )
-    test_loader = DataLoader(
-        test_dataset,
-        shuffle=True,
-        batch_size=args.batch_size,
-        drop_last=True,
-        pin_memory=torch.cuda.is_available() and not args.cpu,
-        num_workers=max(1, available_cpus() // 2),
-        persistent_workers=True,
-    )
+    # Use all CPUs for the training loader during the hot path.  The
+    # test loader is only active at epoch end (full eval) and at
+    # sample_validate_every (quick eval), so a single non-persistent
+    # worker is enough — we don't want it permanently consuming a
+    # CPU slot during training.
+    n_train_workers = max(1, available_cpus())
+    train_loader = DataLoader(
+        train_dataset,
+        shuffle=True,
+        batch_size=args.batch_size,
+        drop_last=True,
+        pin_memory=torch.cuda.is_available() and not args.cpu,
+        num_workers=n_train_workers,
+        persistent_workers=True,
+        prefetch_factor=4,
+    )
+    test_loader = DataLoader(
+        test_dataset,
+        shuffle=True,
+        batch_size=args.batch_size,
+        drop_last=True,
+        pin_memory=torch.cuda.is_available() and not args.cpu,
+        num_workers=1,
+        persistent_workers=False,
+        prefetch_factor=2,
+    )
```

**Rationale on the test loader's `num_workers=1, persistent=False`:**
- Eval is rare relative to train batches (epoch-end + sample_validate_every).
- A non-persistent worker is spawned for the eval pass and destroyed
  after, freeing the CPU slot during training.
- The cost is the worker spawn overhead per eval — but `quick_evaluate`
  samples 10 batches and full eval iterates the test set, both of which
  amortize spawn time easily.

**Edge case:** with `available_cpus() == 8` (the probe), this gives 8
train workers. PyTorch DataLoader supports that fine. The probe was
already achieving ~3.5 min/epoch at 8 CPUs, so this is mostly a fix for
the 2-CPU array runs.

**Test impact:**
- Existing tests construct loaders with `num_workers=0`. Untouched.
- Smoke test (`test_pipeline_integration.py`) runs the real loader code
  path. Should still pass — it doesn't pin worker counts.
- One new smoke test (Change 2.3) verifies the new loader iterates.

### Change 2.2: non-blocking host→device transfer

**Files:**
- `pygv/vampnet/vampnet.py:843-844` (and 1103-1104, 1207-1208 for other
  `to_device` helpers)
- `pygv/vampnet/rev_vampnet.py:602-603` (and 811-812, 887-888)

**Why:** with `pin_memory=True` (already set), `Tensor.to(device,
non_blocking=True)` can overlap H2D copy with previous compute. Without
`non_blocking`, the transfer is synchronous and serializes against the
GPU stream.

**Diff:**

```diff
 def to_device(batch, device_):
     x_t0, x_t1 = batch
-    return (x_t0.to(device_), x_t1.to(device_))
+    return (x_t0.to(device_, non_blocking=True),
+            x_t1.to(device_, non_blocking=True))
```

Apply to all six call sites (three in each of `vampnet.py` and
`rev_vampnet.py`).

**Safety note:** non_blocking is safe only for pinned host memory. The
loaders set `pin_memory=True` in the GPU path; the CPU path
(`args.cpu=True`) sets `pin_memory=False`, in which case `non_blocking`
is silently a no-op. No new branch needed.

**Test impact:** none. Existing tests run on CPU where `non_blocking` is
inert.

### Change 2.3: smoke test for new loader config (optional)

**File:** `tests/test_training.py`

**Why:** ensure the loader changes don't break under
`persistent_workers=True, num_workers>1`. The test harness today uses
`num_workers=0`, so this regime is untested.

**Sketch:**

```python
@pytest.mark.parametrize("n_workers", [0, 2])
def test_loader_iterates_with_workers(synthetic_dataset, n_workers):
    """Loader produces batches under multi-worker config (Tier 2)."""
    from torch_geometric.loader import DataLoader
    loader = DataLoader(
        synthetic_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None,
    )
    batches_seen = 0
    for batch in loader:
        batches_seen += 1
        if batches_seen >= 3:
            break
    assert batches_seen == 3
```

This is a low-value test in isolation but catches obvious pickling/spawn
issues with the dataset class (e.g. lambdas, unpicklable lock state).

---

## Risks and what could go wrong

1. **`distances_self_inf = distances.clone()`**: we clone to avoid
   mutating the original `distances` tensor before computing
   `edge_distances`. This is a small alloc per `__getitem__` call,
   negligible compared to what we removed but worth flagging. (We
   *could* compute `edge_distances` first and skip the clone, but the
   order in the existing code is what it is, and rearranging more is
   higher risk.)

2. **`fill_diagonal_(float('inf'))` + `largest=False`**: standard
   pattern, no UB. `inf` is greater than any finite distance.

3. **Lazy `_rbf_centers` init device**: the first call sets the buffer's
   device from `distances.device`. If the dataset is used on CPU first
   and then moved to GPU later (it isn't — the dataset is CPU-only), we'd
   need a device-aware lookup. Today: not a concern.

4. **Node-feature cache + in-place mutation**: if any caller mutates
   `graph.x` in place, the cache gets corrupted. None do in the repo.
   Worth a comment in the cache helper.

5. **Multi-worker pickling**: `_rbf_centers`, `_rbf_inv_sigma2`,
   `_frames_t`, `_node_attr_cache` are all plain Python attributes
   (tensors and dicts of tensors). All pickle cleanly. The lazy buffer
   pattern means each worker rebuilds its own buffers on first
   `__getitem__`, which is fine — they're identical across workers.

6. **`get_frames_dataset_with_encoding` requesting the opposite
   encoding**: the cache key `(use_amino_acid_encoding,
   amino_acid_feature_type)` handles this — first call computes and
   caches, subsequent calls return the cached tensor.

---

## Test plan summary

### Tests that must still pass unchanged

- `tests/test_dataset.py::TestGraphConstruction::*` — graph shape, edge
  count, no self-loops, deterministic build, valid PyG Data.
- `tests/test_dataset.py::TestNodeFeatures::*` — node feature shapes for
  all encoding modes.
- `tests/test_dataset.py::TestGaussianExpansion::*` — RBF output shape,
  bounds, determinism.
- `tests/test_dataset.py::TestRBFAgainstGhorbaniReference::*` — all six
  tests, including `test_negative_distance_sentinel_zeroed` (the one
  that pins the -1 sentinel behavior).
- `tests/test_dataset.py::TestLagTimeValidation::*`, `TestCaching::*`,
  `TestDatasetInterface::*`, `TestEdgeCases::*` — orthogonal to the
  hot path; should be untouched.
- `tests/test_pipeline_integration.py`, `tests/test_training.py` — the
  end-to-end smoke tests. Should pass with no modification.

### Tests to add

1. **`TestGraphBuildRegression::test_sorted_edge_equivalence`** —
   parametrized over four `(n_atoms, n_neighbors)` shapes covering
   Trp-cage, Villin, Aβ42, and a tiny edge case. Pins
   sorted-edge + edge_attr equivalence between the new vectorized
   implementation and the frozen legacy implementation.
2. **`TestGraphBuildRegression::test_vectorized_edge_ordering_is_deterministic`** —
   pins the row-major source ordering of the new code, so future
   refactors that reintroduce ordering nondeterminism surface
   explicitly.
3. **`test_loader_iterates_with_workers`** (optional, in
   `test_training.py`) — smoke test for `num_workers>0,
   persistent_workers=True`.

### Tests that may need updating

None. Re-read of all existing `test_dataset.py` tests confirms none of
them pin behavior that our changes break:
- They check shapes, bounds, and intra-process determinism — all
  preserved.
- They don't compare raw `edge_index` across processes.
- They don't mutate `graph.x` in-place.

---

## Rollout

1. Land Tier 1 changes together (one PR). The four sub-changes are
   tightly coupled; splitting them creates intermediate states that are
   harder to reason about.
2. Run the existing dataset test suite: `pytest tests/test_dataset.py -v`.
3. Run the new regression test class: `pytest
   tests/test_dataset.py::TestGraphBuildRegression -v`.
4. Run the end-to-end smoke tests: `pytest tests/test_pipeline_integration.py
   tests/test_training.py -v`.
5. Time one Trp-cage v1 seed at 2 CPUs/job (e.g. `sbatch --array=0-0
   cluster_scripts/trpcage_repro_v1_array.sh`). Compare to the 15 min/epoch
   baseline. Target: ≤ 5 min/epoch.
6. If acceptance bar met, land Tier 2 in a follow-up PR (DataLoader
   config + non_blocking). Smaller blast radius; easier to bisect if
   anything regresses.
7. If acceptance bar not met after Tier 1+2, profile to confirm graph
   construction is no longer the bottleneck. If something else dominates
   (covariance/eigh on tiny matrices? edge_attr GPU transfer?),
   investigate that. Pre-building (Option B in the investigation brief)
   is the next step, but should not be needed.

---

## What is intentionally NOT in this plan

- **Cache format changes.** Tier 1+2 keep the pickle layout identical;
  v11 villin caches and any existing Trp-cage caches still load.
- **Option A/B prebuild from the brief.** Held in reserve as Tier 3.
- **AMP / autocast.** GPU-side optimization; only relevant once CPU is
  no longer the bottleneck.
- **Removing `self.node_embeddings` dead code.** Out of scope; track
  separately.
- **Fixing `lag_frames` not being restored from cache.** Out of scope
  (not a perf issue; flagged for a separate cleanup pass).
- **`.item()` syncs in the training loop** (`vampnet.py:903`). GPU-side;
  out of scope for the data-path PR.
