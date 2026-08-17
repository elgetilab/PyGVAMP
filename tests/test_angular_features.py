"""
Angular node features — the cheap pre-test for the PaiNN hypothesis (2026-08-17).

Motivation (claude/PAINN_SCOPE.md §7). GIN and ML3 changed the AGGREGATION over the
same two-body Cα distance inputs and neither beat SchNet on any system. That is
consistent with the binding constraint being the *information* in the descriptor,
not the expressiveness of the network. PaiNN is the first candidate that changes the
inputs — it carries directional/angular structure a distance-only representation
does not contain.

Before spending 30 runs on PaiNN, this adds explicit angular features to the
EXISTING SchNet path and asks whether angular information moves VAMP-2 at all.

These tests pin the two properties the whole argument rests on:
  * the angular features carry information the distance map provably does not
    (test_signed_dihedral_flips_under_reflection +
     test_distance_matrix_is_blind_to_reflection, read as a pair), and
  * they are otherwise rigid-motion invariant, so VAMP states stay well-defined.

If the reflection pair ever goes red, the pre-test is not testing what it claims.
"""

import math
import numpy as np
import pytest
import torch

from pygv.dataset.angular import chain_angular_features, knn_angular_features


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _right_handed_helix(n=12, radius=2.3, rise=1.5, turn=100.0):
    """A right-handed Cα-like helix. Chiral, so reflection is detectable."""
    t = torch.arange(n, dtype=torch.float64)
    ang = torch.deg2rad(torch.tensor(turn, dtype=torch.float64)) * t
    return torch.stack([radius * torch.cos(ang),
                        radius * torch.sin(ang),
                        rise * t], dim=1).float()


def _rotation(axis=(0.3, -0.7, 0.5), angle=0.9):
    """A proper rotation (det = +1) via Rodrigues."""
    k = torch.tensor(axis, dtype=torch.float64)
    k = k / k.norm()
    K = torch.tensor([[0., -k[2], k[1]],
                      [k[2], 0., -k[0]],
                      [-k[1], k[0], 0.]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R.float()


def _reflection():
    """An improper transform (det = -1): mirror through the xy-plane."""
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def _knn_edges(coords, k=7):
    """Same k-NN convention the dataset uses: edge_index = [target, source]."""
    d = torch.cdist(coords, coords)
    d.fill_diagonal_(float('inf'))
    k = min(k, coords.shape[0] - 1)
    _, nn = torch.topk(d, k, dim=1, largest=False)
    src = torch.arange(coords.shape[0]).unsqueeze(1).expand(-1, k).reshape(-1)
    tgt = nn.reshape(-1)
    return torch.stack([tgt, src], dim=0)


# ---------------------------------------------------------------------------
# 1. the pair that justifies the whole experiment
# ---------------------------------------------------------------------------

def test_distance_matrix_is_blind_to_reflection():
    """The baseline descriptor cannot see handedness.

    SchNet/GIN/ML3 all consume Gaussian-expanded pairwise distances. A pairwise
    distance matrix fixes a point set only up to rigid motion INCLUDING reflection,
    so a structure and its mirror image are literally identical inputs.
    """
    coords = _right_handed_helix()
    mirrored = coords @ _reflection().T

    d0 = torch.cdist(coords, coords)
    d1 = torch.cdist(mirrored, mirrored)

    assert torch.allclose(d0, d1, atol=1e-5), (
        "distance matrices differ under reflection — the premise of this "
        "experiment is wrong, re-derive before trusting any result"
    )


def test_signed_dihedral_flips_under_reflection():
    """The angular feature DOES see handedness — this is the information gain.

    Read together with test_distance_matrix_is_blind_to_reflection: cos(tau) is
    preserved (it is reflection-even) while sin(tau) flips sign, so the signed
    dihedral distinguishes structures the distance map cannot.
    """
    coords = _right_handed_helix()
    mirrored = coords @ _reflection().T

    f0 = chain_angular_features(coords)
    f1 = chain_angular_features(mirrored)

    cos_tau0, sin_tau0 = f0[:, 2], f0[:, 3]
    cos_tau1, sin_tau1 = f1[:, 2], f1[:, 3]

    assert torch.allclose(cos_tau0, cos_tau1, atol=1e-5), "cos(tau) must be reflection-even"
    defined = sin_tau0.abs() > 1e-6
    assert defined.any(), "helix should have well-defined dihedrals"
    assert torch.allclose(sin_tau0[defined], -sin_tau1[defined], atol=1e-5), (
        "sin(tau) must flip sign under reflection — without this the angular "
        "features add nothing the distance map lacks"
    )


# ---------------------------------------------------------------------------
# 2. rigid-motion invariance (VAMP states must stay well-defined)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [chain_angular_features, None])
def test_angular_features_are_translation_invariant(fn):
    coords = _right_handed_helix()
    shifted = coords + torch.tensor([13.0, -4.0, 7.5])
    if fn is None:
        e = _knn_edges(coords)
        a, b = knn_angular_features(coords, e), knn_angular_features(shifted, e)
    else:
        a, b = fn(coords), fn(shifted)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("fn", [chain_angular_features, None])
def test_angular_features_are_rotation_invariant(fn):
    coords = _right_handed_helix()
    rotated = coords @ _rotation().T
    if fn is None:
        e = _knn_edges(coords)
        a, b = knn_angular_features(coords, e), knn_angular_features(rotated, e)
    else:
        a, b = fn(coords), fn(rotated)
    assert torch.allclose(a, b, atol=1e-4), (
        "angular features must be rotation-invariant, or the VAMP states would "
        "depend on the arbitrary lab frame of each MD frame"
    )


def test_knn_angular_descriptor_is_reflection_invariant():
    """Documents a real limitation, so it is not mistaken for a bug later.

    cos(angle j-i-l) is reflection-even, so the k-NN descriptor adds angular
    resolution but NOT handedness. Only the signed dihedral carries chirality.
    """
    coords = _right_handed_helix()
    mirrored = coords @ _reflection().T
    e = _knn_edges(coords)
    assert torch.allclose(knn_angular_features(coords, e),
                          knn_angular_features(mirrored, e), atol=1e-4)


# ---------------------------------------------------------------------------
# 3. shape / padding contract
# ---------------------------------------------------------------------------

def test_chain_features_shape_and_end_padding():
    n = 12
    coords = _right_handed_helix(n=n)
    f = chain_angular_features(coords)
    assert f.shape == (n, 4)
    assert torch.isfinite(f).all(), "no NaN/inf may reach the network"
    # theta undefined at both ends; tau undefined for the last three
    assert torch.allclose(f[0, :2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(f[-1, :2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(f[-1, 2:], torch.zeros(2), atol=1e-6)


def test_knn_features_shape_and_finiteness():
    n, bins = 12, 8
    coords = _right_handed_helix(n=n)
    e = _knn_edges(coords, k=7)
    f = knn_angular_features(coords, e, n_bins=bins)
    assert f.shape == (n, bins)
    assert torch.isfinite(f).all()


def test_degenerate_geometry_does_not_produce_nan():
    """Collinear atoms make the dihedral normal vectors vanish — must not NaN."""
    coords = torch.zeros(6, 3)
    coords[:, 2] = torch.arange(6, dtype=torch.float32)  # perfectly collinear
    f = chain_angular_features(coords)
    assert torch.isfinite(f).all(), "collinear geometry produced NaN/inf"


def test_short_chain_shorter_than_dihedral_window():
    """Fewer than 4 atoms: everything undefined, must still return the right shape."""
    for n in (1, 2, 3):
        f = chain_angular_features(_right_handed_helix(n=n))
        assert f.shape == (n, 4)
        assert torch.isfinite(f).all()
        assert torch.allclose(f[:, 2:], torch.zeros(n, 2), atol=1e-6)


# ---------------------------------------------------------------------------
# 4. dataset integration
# ---------------------------------------------------------------------------

from tests.test_dataset import (  # noqa: E402  (fixtures reused deliberately)
    mock_topology, mock_trajectory, mock_mdtraj, dataset_params,
)


@pytest.mark.parametrize("mode,extra", [('none', 0), ('chain', 4), ('knn', 8), ('both', 12)])
def test_node_feature_width_grows_by_the_declared_amount(mock_mdtraj, dataset_params, mode, extra):
    """node_dim is auto-inferred from the dataset (training.py:609), so the encoder
    picks the extra columns up automatically — but only if they actually appear."""
    from pygv.dataset.vampnet_dataset import VAMPNetDataset
    base = VAMPNetDataset(**dataset_params)
    g0, _ = base[0]
    baseline_width = g0.x.shape[1]

    ds = VAMPNetDataset(**dataset_params, angular_features=mode, angular_bins=8)
    g, _ = ds[0]
    assert g.x.shape[1] == baseline_width + extra
    assert torch.isfinite(g.x).all()


def test_default_leaves_the_baseline_descriptor_untouched(mock_mdtraj, dataset_params):
    """The published SchNet/GIN/ML3 numbers must be reproducible bit-for-bit.

    Anything other than an exact match means the pre-test changed the control arm
    and the comparison against the recorded 4.6516 baseline would be invalid.
    """
    from pygv.dataset.vampnet_dataset import VAMPNetDataset
    a, _ = VAMPNetDataset(**dataset_params)[0]
    b, _ = VAMPNetDataset(**dataset_params, angular_features='none')[0]
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert torch.equal(a.edge_index, b.edge_index)


def test_invalid_mode_fails_at_construction_not_mid_run(mock_mdtraj, dataset_params):
    """A typo must not survive until the first __getitem__ three hours in."""
    from pygv.dataset.vampnet_dataset import VAMPNetDataset
    with pytest.raises(ValueError, match="Unknown angular_features mode"):
        VAMPNetDataset(**dataset_params, angular_features='chian')
