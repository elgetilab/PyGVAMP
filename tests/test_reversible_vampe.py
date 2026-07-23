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
    PHASE_CONFIG, apply_phase_freeze,
    matrix_inverse, covariances_E, compute_pi, algebraic_init_us,
)
from pygv.scores.vamp_score_v0 import VAMPScore


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


# --- Three-phase schedule ---------------------------------------------------

def test_phase_config_values():
    assert PHASE_CONFIG['chi'] == {'train_chi': True,  'train_rev': False, 'loss': 'vamp2'}
    assert PHASE_CONFIG['us']  == {'train_chi': False, 'train_rev': True,  'loss': 'vampe'}
    assert PHASE_CONFIG['all'] == {'train_chi': True,  'train_rev': True,  'loss': 'vampe'}


def test_apply_phase_freeze_sets_requires_grad_and_trainable():
    chi = [torch.nn.Parameter(torch.randn(2, 2))]
    rev = [torch.nn.Parameter(torch.randn(2, 2))]

    tr, kind = apply_phase_freeze(chi, rev, 'chi')
    assert chi[0].requires_grad and not rev[0].requires_grad
    assert kind == 'vamp2' and tr == chi

    tr, kind = apply_phase_freeze(chi, rev, 'us')
    assert (not chi[0].requires_grad) and rev[0].requires_grad
    assert kind == 'vampe' and tr == rev

    tr, kind = apply_phase_freeze(chi, rev, 'all')
    assert chi[0].requires_grad and rev[0].requires_grad
    assert kind == 'vampe' and len(tr) == 2

    with pytest.raises(ValueError):
        apply_phase_freeze(chi, rev, 'nope')


def test_phase_gradient_isolation():
    """One optimizer step per phase; only the phase's params may change.

    Uses a toy linear χ + the real VAMPU/VAMPS, so the freeze mechanics are
    exercised end-to-end without needing the GNN encoder.
    """
    torch.manual_seed(0)
    M, d_in, n = 3, 5, 128
    chi_net = torch.nn.Linear(d_in, M)
    vampu = VAMPU(M, activation=torch.exp)
    vamps = VAMPS(M, activation=torch.exp)
    x0, x1 = torch.randn(n, d_in), torch.randn(n, d_in)
    vamp2 = VAMPScore(method='VAMP2')

    chi_params = list(chi_net.parameters())
    rev_params = list(vampu.parameters()) + list(vamps.parameters())

    for phase in ['chi', 'us', 'all']:
        trainable, kind = apply_phase_freeze(chi_params, rev_params, phase)
        opt = torch.optim.Adam(trainable, lr=0.1)
        chi_before = [p.detach().clone() for p in chi_params]
        rev_before = [p.detach().clone() for p in rev_params]

        opt.zero_grad()
        c0, c1 = torch.softmax(chi_net(x0), 1), torch.softmax(chi_net(x1), 1)
        if kind == 'vamp2':
            loss = vamp2.loss(c0, c1)
        else:
            u, v, C00, C11, C01, sigma, mu = vampu(c0, c1)
            ve, K, S = vamps(v, C00, C11, C01, sigma)
            loss = vampe_trace_loss(ve)
        loss.backward()
        opt.step()

        chi_changed = any(not torch.allclose(a, b) for a, b in zip(chi_before, chi_params))
        rev_changed = any(not torch.allclose(a, b) for a, b in zip(rev_before, rev_params))
        cfg = PHASE_CONFIG[phase]
        assert chi_changed == cfg['train_chi'], f"phase {phase}: χ update mismatch"
        assert rev_changed == cfg['train_rev'], f"phase {phase}: reversible update mismatch"


def test_three_phase_dispatch_requires_epoch_args():
    """The pipeline wiring errors clearly if --rev_three_phase lacks phase epochs."""
    from types import SimpleNamespace
    from pygv.pipe.training import _train_reversible_three_phase
    args = SimpleNamespace(
        rev_activation='exp', epoch_chi=None, epoch_us=10, epoch_all=10,
        lr_chi=None, lr_us=None, lr_all=None, weight_decay=1e-5,
        n_states=3, rev_renorm=False,
    )
    with pytest.raises(ValueError, match="epoch_chi"):
        _train_reversible_three_phase(args, model=None, train_loader=None,
                                      test_loader=None, paths={'model_dir': '/tmp'},
                                      device='cpu')


# --- Algebraic U/S init (RevGraphVAMP Stage 2) -----------------------------

def test_matrix_inverse_matches_numpy_inv():
    import numpy as np
    torch.manual_seed(0)
    A = torch.randn(5, 5)
    spd = A @ A.t() + 3.0 * torch.eye(5)   # well-conditioned SPD
    inv = matrix_inverse(spd)
    np.testing.assert_allclose(inv, np.linalg.inv(spd.numpy()), rtol=1e-4, atol=1e-5)


def test_compute_pi_recovers_stationary():
    import numpy as np
    # K row-stochastic with known stationary pi = [2/3, 1/3].
    K = np.array([[0.8, 0.2], [0.4, 0.6]])
    pi = compute_pi(K)
    np.testing.assert_allclose(pi, [2.0 / 3.0, 1.0 / 3.0], atol=1e-6)
    # pi is a left eigenvector for eigenvalue 1: pi @ K == pi
    np.testing.assert_allclose(pi @ K, pi, atol=1e-6)


def test_covariances_E_reconstructs_koopman():
    import numpy as np
    chi0, chi1 = _random_chi(n_batch=500, M=3, seed=7)
    C0inv, Ctau = covariances_E(chi0, chi1)
    K = C0inv @ Ctau
    # Manual: K = C00^{-1} C01 with non-mean-removed covariances.
    c0 = chi0.numpy(); c1 = chi1.numpy(); n = c0.shape[0]
    C00 = (c0.T @ c0) / n
    C01 = (c0.T @ c1) / n
    np.testing.assert_allclose(K, np.linalg.inv(C00) @ C01, rtol=1e-3, atol=1e-4)


def test_algebraic_init_recovers_reversible_vampe_on_metastable_data():
    """On data with real metastability, the closed-form init should lift the
    reversible VAMP-E from its (bad) default up to near the standard VAMP-E
    ceiling — the whole point of RevGraphVAMP Stage 2. (On structureless/random
    chi there is nothing to fit, so this is tested on a genuine 2-state process.)
    """
    import numpy as np
    rng = np.random.RandomState(0)
    N, p_stay = 20000, 0.99                 # slow 2-state Markov chain
    s = np.zeros(N, dtype=int)
    for i in range(1, N):
        s[i] = s[i - 1] if rng.rand() < p_stay else 1 - s[i - 1]

    def to_chi(states):                     # near one-hot softmax features + noise
        logits = np.zeros((len(states), 2))
        logits[np.arange(len(states)), states] = 3.0
        logits += 0.5 * rng.randn(len(states), 2)
        e = np.exp(logits)
        return (e / e.sum(1, keepdims=True)).astype(np.float32)

    chi0 = torch.from_numpy(to_chi(s[:-1]))
    chi1 = torch.from_numpy(to_chi(s[1:]))
    vampu, vamps = VAMPU(2, activation=torch.exp), VAMPS(2, activation=torch.exp)

    def rev_vampe():
        u, v, C00, C11, C01, sig, mu = vampu(chi0, chi1)
        ve, K, S = vamps(v, C00, C11, C01, sig)
        return reversible_vampe_score(ve).item()

    before = rev_vampe()
    algebraic_init_us(vampu, vamps, chi0, chi1)
    after = rev_vampe()
    ceiling = VAMPScore(method='VAMPE')(chi0, chi1).item()

    assert after > before, f"init did not improve VAMP-E: {before} -> {after}"
    assert after > ceiling - 0.3, f"init VAMP-E {after} far below ceiling {ceiling}"
    assert torch.isfinite(vampu._u_kernel).all()
    assert torch.isfinite(vamps._s_kernel).all()
