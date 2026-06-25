"""Regression tests for temporally blocked train/val splitting.

Covers the two properties that make the blocked split *honest*:
  1. train and val pair sets are disjoint, and
  2. no kept pair straddles a train<->val block seam (the lag-width buffer).
"""
import numpy as np

from pygv.dataset.splits import make_blocked_split, make_random_split


def _continuous_pairs(n_frames, lag):
    t0 = list(range(n_frames - lag))
    t1 = list(range(lag, n_frames))
    return t0, t1


def test_blocked_split_disjoint_and_buffered():
    n_frames, lag = 10_000, 100
    t0, t1 = _continuous_pairs(n_frames, lag)
    train_idx, val_idx, n_dropped = make_blocked_split(
        t0, t1, n_frames, val_frac=0.3, n_blocks=10, seed=0)

    # Disjoint, and together with the dropped seam pairs they cover all pairs.
    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(train_idx) + len(val_idx) + n_dropped == len(t0)
    assert len(val_idx) > 0 and len(train_idx) > 0

    # No kept pair straddles a seam: rebuild the block / val-block assignment and
    # verify BOTH endpoints of every kept pair fall on the same (val or train) side.
    edges = np.linspace(0, n_frames, 11).astype(int)
    frame_block = np.clip(np.searchsorted(edges, np.arange(n_frames), 'right') - 1, 0, 9)
    t0a, t1a = np.asarray(t0), np.asarray(t1)
    val_blocks = set(frame_block[t0a[val_idx]].tolist())
    assert all(frame_block[t0a[i]] in val_blocks and frame_block[t1a[i]] in val_blocks
               for i in val_idx)
    assert all(frame_block[t0a[i]] not in val_blocks and frame_block[t1a[i]] not in val_blocks
               for i in train_idx)
    # Seam buffer actually dropped something (lag spans block boundaries).
    assert n_dropped > 0


def test_blocked_val_fraction_in_range():
    n_frames, lag = 50_000, 100
    t0, t1 = _continuous_pairs(n_frames, lag)
    _, val_idx, _ = make_blocked_split(t0, t1, n_frames, val_frac=0.3, n_blocks=10, seed=1)
    frac = len(val_idx) / len(t0)
    # 3 of 10 blocks -> ~0.3, minus seam drops; allow a generous band.
    assert 0.2 < frac < 0.32


def test_random_split_disjoint_and_sized():
    n = 1000
    train_idx, val_idx = make_random_split(n, val_frac=0.3, seed=0)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(val_idx) == 300 and len(train_idx) == 700


def test_blocked_split_is_reproducible():
    n_frames, lag = 5000, 50
    t0, t1 = _continuous_pairs(n_frames, lag)
    a = make_blocked_split(t0, t1, n_frames, seed=7)
    b = make_blocked_split(t0, t1, n_frames, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
