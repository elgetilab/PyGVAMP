"""
Correctness tests for the reversible VAMP-E layer (VAMPU / VAMPS).

Pins the ported RevGraphVAMP / deep_rev_msm math against properties that are
derivable independently of the port itself, so a transcription error would fail:

1. ``S @ v == 1`` — the deep_rev_msm constraint the ``w2 = (1 - w_norm)/v`` line
   enforces (the trickiest line; this is the golden check).
2. ``S`` symmetric.
3. ``vamp_e`` equals ``Sᵀ C00 S C11 − 2 Sᵀ C01`` recomputed from the module's own
   S and covariances (catches wiring/transcription bugs).
4. Gradients flow to both ``_u_kernel`` and ``_s_kernel`` through ``trace(vamp_e)``.
5. The numerical guards clamp a degenerate ``v`` (near-zero entry) without NaN,
   and warn when they fire.

Run: pytest tests/test_reversible_vampe.py -v
"""

import warnings

import numpy as np
import pytest
import torch

from pygv.scores.reversible_vampe import (
    VAMPU, VAMPS, vampe_trace_loss, reversible_vampe_score, _clamp_away_from_zero,
)


def _random_chi(n_batch=256, M=3, seed=0):
    """A pair of softmax feature matrices (chi_t, chi_tau)."""
    g = torch.Generator().manual_seed(seed)
    logits_t = torch.randn(n_batch, M, generator=g)
    logits_tau = logits_t + 0.3 * torch.randn(n_batch, M, generator=g)  # correlated
    return torch.softmax(logits_t, dim=1), torch.softmax(logits_tau, dim=1)


def _run_layer(M=3, seed=0, renorm=False):
    chi_t, chi_tau = _random_chi(M=M, seed=seed)
    vampu = VAMPU(M, activation=torch.exp)
    vamps = VAMPS(M, activation=torch.exp, renorm=renorm)
    u, v, C00, C11, C01, sigma, mu = vampu(chi_t, chi_tau)
    vamp_e, K, S = vamps(v, C00, C11, C01, sigma)
    return dict(v=v, C00=C00, C11=C11, C01=C01, sigma=sigma,
               vamp_e=vamp_e, K=K, S=S, vampu=vampu, vamps=vamps)


@pytest.mark.parametrize("M", [2, 3, 5])
def test_S_times_v_is_ones(M):
    # The golden check: the w2 constraint forces S @ v = 1 exactly.
    r = _run_layer(M=M, seed=M)
    Sv = r["S"] @ r["v"].reshape(-1)
    assert torch.allclose(Sv, torch.ones(M), atol=1e-4), Sv


def test_S_symmetric():
    r = _run_layer(M=4, seed=1)
    assert torch.allclose(r["S"], r["S"].t(), atol=1e-5)


def test_vamp_e_formula_matches_recompute():
    r = _run_layer(M=3, seed=2)
    S, C00, C11, C01 = r["S"], r["C00"], r["C11"], r["C01"]
    expected = S.t() @ C00 @ S @ C11 - 2.0 * S.t() @ C01
    assert torch.allclose(r["vamp_e"], expected, atol=1e-6)
    # score == -trace(vamp_e); loss == trace(vamp_e)
    assert torch.allclose(vampe_trace_loss(r["vamp_e"]), torch.trace(r["vamp_e"]))
    assert torch.allclose(reversible_vampe_score(r["vamp_e"]), -torch.trace(r["vamp_e"]))


def test_gradients_flow_to_both_kernels():
    chi_t, chi_tau = _random_chi(M=3, seed=3)
    vampu = VAMPU(3, activation=torch.exp)
    vamps = VAMPS(3, activation=torch.exp)
    u, v, C00, C11, C01, sigma, mu = vampu(chi_t, chi_tau)
    vamp_e, K, S = vamps(v, C00, C11, C01, sigma)
    loss = vampe_trace_loss(vamp_e)
    loss.backward()
    assert vampu._u_kernel.grad is not None and torch.isfinite(vampu._u_kernel.grad).all()
    assert vamps._s_kernel.grad is not None and torch.isfinite(vamps._s_kernel.grad).all()


def test_clamp_helper_preserves_sign_and_warns():
    x = torch.tensor([1.0, 1e-9, -1e-9, -2.0])
    with pytest.warns(RuntimeWarning):
        y = _clamp_away_from_zero(x, eps=1e-6, name="test")
    assert y[0] == pytest.approx(1.0)
    assert y[3] == pytest.approx(-2.0)
    assert y[1] == pytest.approx(1e-6)      # sign preserved (was +)
    assert y[2] == pytest.approx(-1e-6)     # sign preserved (was -)


def test_guard_on_degenerate_v_no_nan():
    # Feed VAMPS a v with a near-zero entry directly; assert finite S, no NaN.
    M = 3
    vamps = VAMPS(M, activation=torch.exp)
    v = torch.tensor([0.5, 1e-12, 0.4])          # one degenerate entry
    C = torch.eye(M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vamp_e, K, S = vamps(v, C, C, C, C)
    assert torch.isfinite(S).all()
    assert torch.isfinite(vamp_e).all()
