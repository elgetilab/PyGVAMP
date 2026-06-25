"""Train/validation partitioning for time-lagged VAMPNet datasets.

The default pipeline uses ``torch.utils.data.random_split`` over the time-lagged
pair dataset (see ``pygv/pipe/training.py``).  Because consecutive MD frames are
highly autocorrelated, a *random* (interleaved) split places validation pairs
``(x_t, x_{t+τ})`` next to near-identical training configurations — a temporal
leak that makes the validation VAMP-2 an optimistic estimate.

This module adds a **temporally blocked** split: whole contiguous frame blocks go
to train or validation, blocks are spread across the trajectory for
representativeness, and any pair whose ``[t0, t1]`` window straddles a
train<->val block seam is *dropped* (a lag-width buffer) so no information leaks
across the boundary.

Both helpers return **pair indices** into the pair dataset (i.e. indices into the
``t0_indices``/``t1_indices`` lists), suitable for ``torch.utils.data.Subset``.

NOTE (trajectory boundaries): blocks are defined over the *concatenated* frame
index space.  This is exactly correct for a single physical trajectory (e.g. the
Trp-cage 2JOF run split across .dcd segments, ``continuous=True``).  For a dataset
of several *independent* trajectories (``continuous=False``), pairs already never
cross a trajectory boundary, but a block edge could fall mid-trajectory; the seam
buffer still prevents leakage, though block boundaries are not snapped to
trajectory edges.  Snapping is a future refinement; not needed for the Trp-cage
split-leakage tests.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def make_random_split(n_pairs: int, val_frac: float = 0.3, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Interleaved random split over pair indices (reproduces the leaky baseline).

    Parameters
    ----------
    n_pairs : int
        Number of time-lagged pairs (``len(dataset)``).
    val_frac : float
        Fraction of pairs held out for validation.
    seed : int
        RNG seed.

    Returns
    -------
    (train_idx, val_idx) : tuple of np.ndarray
        Sorted pair indices.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pairs)
    n_val = int(n_pairs * val_frac)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def make_blocked_split(t0_indices, t1_indices, n_frames: int,
                       val_frac: float = 0.3, n_blocks: int = 10, seed: int = 0,
                       trajectory_boundaries: Optional[list] = None
                       ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Temporally blocked split over pair indices, with a lag-width seam buffer.

    Each pair is binned by the block of its *start* frame ``t0``.  ``val_frac`` of
    the blocks (spread across the trajectory) are assigned to validation; the rest
    to training.  Any pair whose ``[t0, t1]`` window straddles a train<->val seam
    is dropped.

    Parameters
    ----------
    t0_indices, t1_indices : sequence of int
        Pair start/end frame indices (``dataset.t0_indices`` / ``.t1_indices``).
    n_frames : int
        Total number of frames the indices reference (``dataset.n_frames``).
    val_frac : float
        Target fraction of *blocks* assigned to validation.
    n_blocks : int
        Number of contiguous blocks to partition the trajectory into.
    seed : int
        RNG seed for choosing which blocks are validation.
    trajectory_boundaries : list, optional
        Frame start indices per trajectory (reserved; see module note).

    Returns
    -------
    (train_idx, val_idx, n_dropped) : tuple
        Sorted pair indices for train and val, and the number of pairs dropped at
        seams.
    """
    t0 = np.asarray(t0_indices, dtype=np.int64)
    t1 = np.asarray(t1_indices, dtype=np.int64)
    if t0.shape != t1.shape:
        raise ValueError("t0_indices and t1_indices must have the same length")
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 for a blocked split")

    rng = np.random.default_rng(seed)

    # Contiguous frame -> block map.
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(np.int64)
    frame_block = np.searchsorted(edges, np.arange(n_frames), side='right') - 1
    np.clip(frame_block, 0, n_blocks - 1, out=frame_block)

    # Choose validation blocks, spread across the trajectory.
    n_val_blocks = max(1, round(n_blocks * val_frac))
    val_blocks = rng.choice(n_blocks, size=n_val_blocks, replace=False)
    is_val_block = np.zeros(n_blocks, dtype=bool)
    is_val_block[val_blocks] = True

    in_val_t0 = is_val_block[frame_block[t0]]
    in_val_t1 = is_val_block[frame_block[t1]]

    straddle = in_val_t0 != in_val_t1          # crosses a train<->val seam -> drop
    keep = ~straddle
    val_idx = np.nonzero(keep & in_val_t0)[0]
    train_idx = np.nonzero(keep & ~in_val_t0)[0]
    n_dropped = int(straddle.sum())

    return np.sort(train_idx), np.sort(val_idx), n_dropped
