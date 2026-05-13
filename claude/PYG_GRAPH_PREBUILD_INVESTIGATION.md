# PyG graph pre-building — investigation brief

**Status:** open investigation, separate from any in-flight experiment.
Hand this brief to a fresh Claude Code process.

**Goal:** decide whether pre-building PyG `Data` objects (or at least
their `edge_index` / `edge_attr` / `x` tensors) into the on-disk cache
is a worthwhile speed-up for the small-architecture reproduction
workloads, and if so, implement it.

---

## Why this matters

The Trp-cage v1 10-seed array (job 522, 2026-05-12) revealed the
small-architecture reproduction protocol is severely **CPU-bound on
graph construction**, not GPU-bound on training compute.

| Configuration | Per-epoch wall time | GPU 0 utilization |
|---|---|---|
| Probe (1 job × 8 CPUs)        | ~3.5 min/epoch | not measured (likely ~10–20%) |
| Array (8 jobs × 2 CPUs each)  | ~15 min/epoch | **5% combined** |

The 4× slowdown when dropping CPUs from 8 → 2 is almost linear, which
points to per-batch work in DataLoader workers as the bottleneck.

VRAM use across 8 concurrent jobs: ~12 GB / 32 GB on GPU 0. The card
is essentially idle.

**Implication:** any optimization that moves per-sample graph
construction work *out of the training hot path* (into a one-time
preprocessing step) should multiply throughput by a large factor for
this class of reproduction runs. Larger models (encoder sweeps,
Aβ42 reversible) likely don't benefit since they're GPU-bound — but
those aren't the slow ones.

---

## Where the work actually happens

**File:** `pygv/dataset/vampnet_dataset.py`

**Function:** `_create_graph_from_frame(frame_idx)` — lines ~395–469.

Called **twice per `__getitem__`** (one for `t0`, one for `t1`,
lines 475–483). For Trp-cage, that's `2 × 730 batches/epoch × 1000
samples/batch × 100 epochs = 1.46 × 10⁸ graph builds per seed`.

Each call does, **per frame, every time**:

1. Pairwise distance matrix (cheap — `N=20`).
2. **Python `for` loop over atoms** to find each node's k-nearest
   neighbors (lines 428–437). Uses `torch.topk` inside a Python loop.
3. **Python `set` accumulation** of `(i, j)` edge tuples (lines
   440–443).
4. **Python list comprehensions** to split edges back into source /
   target tensors (lines 446–449).
5. `_compute_gaussian_expanded_distances(edge_distances)` — RBF
   expansion of edge distances (line 456).
6. `_create_node_features(...)` — likely amino-acid one-hot or
   identity features (line 459). **Probably frame-invariant**, worth
   confirming.

Steps 1–5 are deterministic given the frame coordinates. Step 6 may
or may not be frame-dependent depending on `use_amino_acid_encoding`
and the encoder type — verify before caching.

The cache today (`_save_to_cache`, line ~511) pickles `self.frames`,
`self.t0_indices`, `self.t1_indices`, and `self.trajectory_files`.
Cache filename encodes lag/nn/stride/continuity:

```
vampnet_data_{traj_hash}_lag{lag}_nn{nn}_str{stride}_{cont|noncont}.pkl
```

`n_neighbors` is in the cache key, so adding precomputed kNN edges
doesn't break the key contract — the same cache file already
corresponds to a fixed `n_neighbors`.

---

## Investigation questions (answer in order)

1. **Profile-confirm the bottleneck.** Pick one Trp-cage v1 seed, run
   for ~3 epochs with `cProfile` or `py-spy --duration 60` attached to
   a DataLoader worker process. Quantify the share of wall-clock time
   spent inside `_create_graph_from_frame` vs the rest of the
   training loop. Expected: > 70%. If much less, the hypothesis is
   wrong and stop.

2. **Is `_create_node_features` frame-dependent or frame-invariant?**
   Read its body. For Cα-only with `--no_use_embedding`, it's almost
   certainly identical across frames (uniform feature vector). For
   Aβ42 with amino-acid encoding it's also frame-invariant (residue
   identity doesn't change). If frame-invariant, cache one copy.

3. **Are edges frame-dependent?** Yes — kNN connectivity depends on
   per-frame coordinates. So edges must be cached per-frame.

4. **What's the memory cost of caching `(edge_index, edge_attr)` for
   every frame?**
   - Trp-cage: 20 atoms × 7 neighbors ≈ 140 directed edges/frame.
     `edge_index`: `2 × 140 × int64 = 2.2 KB`.
     `edge_attr`: `140 × G × float32` where `G = gaussian_expansion_dim
     = 16` → 8.8 KB.
     Per frame total: ~11 KB. × 1,044,000 frames = **~11 GB**.
     That's larger than the current cache pickle by ~10×. May push
     dataset load time up too.
   - Villin: 35 × 10 = 350 edges/frame. ~26 KB/frame × 1.25M frames =
     ~33 GB.
   - Aβ42: probably similar order of magnitude.
   - Likely fits on `/mnt/hdd` (2 TB user quota) but is not free.

5. **Alternative: pre-compute `edge_index` only, recompute `edge_attr`
   on the fly.** edges are the loop-heavy step. Gaussian expansion is
   pure tensor work, much faster. Trade off some speed for ~10× less
   cache size.

---

## Suggested implementation (only after Q1–Q4 are answered)

**Option A — full pre-build (largest cache, fastest training):**

- In `_save_to_cache`, after `self.frames` is populated, run
  `_create_graph_from_frame` for every frame and store results in
  parallel numpy arrays (`edge_index_all`, `edge_attr_all`,
  `node_attr` once).
- In `__getitem__`, look up by index, wrap in `Data(...)`. Skip the
  per-frame loop entirely.
- New cache file version: bump filename or add a version field to
  the pickle. Old caches without precomputed graphs should still load
  (back-compat — important since v11 villin caches exist on disk).

**Option B — edges only:**

- Cache `edge_index` per frame; recompute `edge_attr` at `__getitem__`
  time via tensor ops on `self.frames[idx]` distances + the cached
  `edge_index`. Skips the slow Python loops; keeps the tensor work.

**Option C — JIT/vectorize the kNN loop:**

- Replace the per-atom Python loop with a single batched `torch.topk`
  on the full distance matrix. This is a small surgery — no cache
  change at all. Likely captures most of the gain and is the
  lowest-risk option to try first.

Recommendation: **start with C** (Q1's profile will confirm whether
even C is enough). If C closes the gap, ship it and stop. Pre-building
is the heavier intervention.

---

## Constraints / non-goals

- **Do not change reproduction behavior.** Same graphs, same edges,
  same edge_attr. This is a pure speed-up. Add a regression test:
  `_create_graph_from_frame` before vs after the change must produce
  identical `Data` objects on a few representative frames (per
  `feedback_test_before_change` — pin behavior with a test first).
- **Do not break existing caches.** v11 villin caches are valuable
  (each took ~10 min to build). Either keep the old format readable
  or detect old caches and fall back gracefully.
- **Do not add CLI flags for this.** It should be a transparent
  performance fix, opt-in only via the existing `--cache` flag.
- **Stay in scope:** only `pygv/dataset/vampnet_dataset.py` and
  matching tests. Do not touch encoders, training loop, or analysis.

---

## Acceptance criteria

- A single seed of Trp-cage v1 with 2 CPUs/job runs at **≤ 5
  min/epoch** (down from 15 min) — 3× speed-up is the minimum bar.
- Existing tests still pass: `pytest tests/test_vampnet_dataset*.py`
  (or whatever the dataset tests are named).
- New test pins `Data` object equivalence (edge_index, edge_attr, x)
  pre- vs post-optimization on a small fixture.
- Old caches still load without error.
- Cache size growth on disk (if Option A) is documented in the PR
  description.

---

## Useful starting commands

```bash
# Locate the dataset module and call sites
grep -nE "def __getitem__|def _create_graph|knn|topk|edge_index" \
    pygv/dataset/vampnet_dataset.py

# Find DataLoader construction sites
grep -rn "DataLoader\|num_workers" pygv/

# Profile one Trp-cage seed for a few epochs (after the array
# finishes; do not run this while job 522 is active).
# Use py-spy attached to a worker PID for live sampling.

# Run existing dataset tests
pytest tests/ -k dataset -v
```

---

## Out of scope here, but adjacent

- DataLoader `num_workers` tuning — separate small win.
- Disk I/O contention across N concurrent jobs reading the same
  cache pickle — only matters at high concurrency.
- Batch-size sensitivity — increasing `batch_size` from 1000 to 4000
  would reduce per-epoch batch count 4×, but changes the optimization
  (different SGD noise) and is **reproduction-incompatible** with
  Ghorbani 2022. Don't touch.
