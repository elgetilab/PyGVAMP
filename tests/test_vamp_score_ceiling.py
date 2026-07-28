"""Regression tests: the VAMP-2 estimator must respect its theoretical ceiling.

Written 2026-07-28 after the Aβ42 10-seed run (job 838) reported a cross-seed
VAMP-2 of 4.0974 at k=4 — above the k=4 ceiling of 4.0, i.e. an impossible score.
These tests pin the invariant BEFORE any fix, so the magnitude of the defect is
recorded rather than argued about.

The maths
---------
The whitened Koopman operator ``K = C00^{-1/2} C0t Ctt^{-1/2}`` is a correlation
operator: every singular value satisfies ``sigma_i <= 1``. The score is
``VAMP-2 = 1 + ||K||_F^2 = 1 + sum_i sigma_i^2``.

For a k-state softmax chi, the rows sum to 1, so the all-ones direction lies
exactly in the null space of the MEAN-REMOVED covariance. C00 therefore has rank
<= k-1, leaving at most k-1 singular values, hence::

    VAMP-2 <= 1 + (k - 1) = k

The defect
----------
``VAMPScore`` defaults to ``mode='trunc'`` with ``epsilon=1e-6`` and drops
eigenvalues with ``eigval > epsilon``. That structurally-zero eigenvalue is not
exactly zero in float32 — it is float noise whose magnitude scales with ||C00||
and with the number of frames accumulated. When it lands just ABOVE 1e-6 it is
RETAINED, and ``_sym_inverse(..., return_sqrt=True)`` then multiplies that
direction by ``1/sqrt(1e-6) ~= 1000``. The resulting spurious singular values
exceed 1 and the score breaks its ceiling.

This is not hypothetical: a confident classifier (which is exactly what the joint
VAMP-E phase produces) at validation-set scale triggers it on every seed tried.
"""

import pytest
import torch

from pygv.scores.vamp_score_v0 import VAMPScore

# Validation-set scale of the Aβ42 run: ~378k frames (30% of 1.26M).
N_VAL = 378_000


def _chi_pair(n, k, sharpness, seed=0):
    """A (chi_t, chi_tau) softmax pair. `sharpness` sets classifier confidence."""
    g = torch.Generator().manual_seed(seed)
    logits = sharpness * torch.randn(n, k, generator=g)
    lagged = logits + 0.05 * sharpness * torch.randn(n, k, generator=g)
    return torch.softmax(logits, dim=1), torch.softmax(lagged, dim=1)


@pytest.mark.parametrize("k", [4, 6])
def test_vamp2_within_ceiling_for_diffuse_chi(k):
    """Sanity: a low-confidence chi scores below the ceiling (this already passes)."""
    c0, c1 = _chi_pair(20_000, k, sharpness=1.0)
    score = VAMPScore(method="VAMP2")(c0, c1).item()
    assert score <= k + 1e-4, f"VAMP-2 {score:.4f} exceeds the k={k} ceiling"


@pytest.mark.parametrize("k,sharpness", [(4, 4.0), (4, 10.0), (6, 4.0)])
def test_vamp2_within_ceiling_for_confident_chi(k, sharpness):
    """A confident chi at validation scale must still respect the ceiling.

    This is the regression case: the joint VAMP-E phase drives chi toward
    near-one-hot assignments, which is precisely when the mean-removed C00's
    structural null eigenvalue drifts above the 1e-6 truncation threshold.
    """
    c0, c1 = _chi_pair(N_VAL, k, sharpness=sharpness)
    score = VAMPScore(method="VAMP2")(c0, c1).item()
    assert score <= k + 1e-4, (
        f"VAMP-2 {score:.4f} exceeds the k={k} ceiling by {score - k:.4f} "
        f"(sharpness={sharpness}) — the estimator is reporting an impossible score"
    )


@pytest.mark.parametrize("sharpness", [1.0, 4.0, 10.0])
def test_whitened_koopman_singular_values_bounded_by_one(sharpness):
    """The root invariant: every singular value of the whitened Koopman is <= 1.

    Asserting this directly localises the defect to the whitening, independently
    of how the score is later assembled.
    """
    k = 4
    c0, c1 = _chi_pair(N_VAL, k, sharpness=sharpness)
    scorer = VAMPScore(method="VAMP2")
    koopman = scorer._koopman_matrix(c0, c1)
    smax = torch.linalg.svdvals(koopman.double()).max().item()
    assert smax <= 1.0 + 1e-4, (
        f"largest singular value {smax:.4f} > 1 (sharpness={sharpness}); "
        f"whitening amplified a truncated-null direction"
    )


def test_structural_null_eigenvalue_is_near_the_truncation_threshold():
    """Documents the trigger: the structural null eigenvalue sits near epsilon=1e-6.

    Softmax rows sum to 1, so the mean-removed C00 is rank-deficient by exactly
    one. That eigenvalue should be treated as zero, but in float32 it is noise of
    order 1e-6 — the same order as the truncation cutoff, so whether it is kept is
    a coin flip. This test does not assert a fix; it pins WHY the ceiling breaks.
    """
    scorer = VAMPScore(method="VAMP2")
    c0, c1 = _chi_pair(N_VAL, 4, sharpness=4.0)
    c00, _, _ = scorer._covariances(c0, c1, remove_mean=True)
    eigs = torch.linalg.eigvalsh(c00.double())
    assert abs(eigs.min().item()) < 1e-3, "expected a near-zero (structural) eigenvalue"
    assert eigs.min().item() < eigs[1].item() / 100, (
        "expected the smallest eigenvalue to be orders below the next one"
    )
