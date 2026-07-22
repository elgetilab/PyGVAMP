"""
Reversible VAMP-E layer (VAMPU / VAMPS) — faithful port of RevGraphVAMP
(Huang et al. 2024, github.com/DS00HY/RevGraphVamp, src/revvamp.py), which is
itself a port of Mardt & Noé's deep reversible MSM (markovmodel/deep_rev_msm).

This is the reproduction-grade reversible training path. It is DISTINCT from
``reversible_score.py`` (the older softplus-K NLL score), which is kept for
backward compatibility and is NOT comparable to RevGraphVAMP's numbers.

Construction (verbatim reference in claude/REVGRAPHVAMP_TODO.md §"Reference code"):

- VAMPU builds the stationary weights ``u`` and the reweighted covariances
  ``C00, C11, C01, sigma`` from the softmax features ``chi_t, chi_tau``.
- VAMPS builds a symmetric ``S`` constrained so that ``S @ v = 1``, the Koopman
  ``K = S @ sigma``, and the VAMP-E matrix
  ``vamp_e = Sᵀ C00 S C11 − 2 Sᵀ C01``.
- Training minimizes ``trace(vamp_e)`` (== maximizing the reversible-K VAMP-E
  score). The reported VAMP-E for the paper table is computed separately via the
  standard ``VAMPScore(method='VAMPE')`` path (verified identical to theirs,
  including the ``1 + out`` constant).

Two lines in the reference are numerically fragile (see REVGRAPHVAMP_TODO.md §4);
both are guarded here with an epsilon clamp that preserves sign and warns when it
fires, so a genuine state collapse is distinguishable from a numerical blow-up.
"""

import warnings

import torch
import torch.nn as nn


def _clamp_away_from_zero(x: torch.Tensor, eps: float, name: str) -> torch.Tensor:
    """Clamp |x| up to ``eps`` while preserving sign; warn if it fires.

    Guards the two fragile reference lines (VAMPU's u-normalization denominator
    and VAMPS's ``(1 - w_norm) / v`` division), where a near-zero entry would
    otherwise send the reversible parameters to infinity.
    """
    small = x.abs() < eps
    n_small = int(small.sum())
    if n_small:
        warnings.warn(
            f"reversible_vampe: {name}: {n_small} entry(ies) within eps={eps} of "
            f"zero were clamped (possible numerical instability, not necessarily "
            f"a state collapse).",
            RuntimeWarning,
        )
        sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        x = sign * torch.clamp(x.abs(), min=eps)
    return x


class VAMPU(nn.Module):
    """Stationary-distribution (u) module of the reversible VAMP-E layer.

    Parameters
    ----------
    units : int
        Number of metastable states (M).
    activation : callable, default ``torch.exp``
        Applied to the learnable kernel to keep it positive. RevGraphVAMP /
        deep_rev_msm use ``exp``. (Confirm against their train script before a
        strict run — tracked as an open item in REVGRAPHVAMP_TODO.md.)
    eps : float
        Numerical guard threshold.
    """

    def __init__(self, units: int, activation=torch.exp, eps: float = 1e-6):
        super().__init__()
        self.M = units
        self.activation = activation
        self.eps = eps
        # Reference init: (1/M) * ones(M)
        self._u_kernel = nn.Parameter((1.0 / units) * torch.ones(units))

    def forward(self, chi_t: torch.Tensor, chi_tau: torch.Tensor):
        """
        Parameters
        ----------
        chi_t, chi_tau : torch.Tensor, shape (n_batch, M)
            Softmax features at time t and t+tau.

        Returns
        -------
        (u, v, C00, C11, C01, sigma, mu)
            ``u`` shape (1, M); ``v`` shape (M, 1); covariances (M, M);
            ``mu`` shape (n_batch, 1). Matrices are returned un-tiled (the
            reference tiles them over the batch dim purely for its loss
            reduction; ``trace(vamp_e)`` is batch-independent, so tiling is a
            no-op here).
        """
        n = chi_t.shape[0]
        norm = 1.0 / n

        corr_tau = norm * (chi_tau.t() @ chi_tau)                 # (M, M)
        chi_mean = chi_tau.mean(dim=0, keepdim=True)              # (1, M)
        kernel_u = self.activation(self._u_kernel).unsqueeze(0)   # (1, M)

        denom = (chi_mean * kernel_u).sum(dim=1, keepdim=True)    # (1, 1)
        denom = _clamp_away_from_zero(denom, self.eps, "u-normalization denominator")
        u = kernel_u / denom                                      # (1, M)
        u_t = u.t()                                               # (M, 1)

        v = corr_tau @ u_t                                        # (M, 1)
        mu = norm * (chi_tau @ u_t)                               # (n, 1)
        sigma = (chi_tau * mu).t() @ chi_tau                      # (M, M)
        gamma = chi_tau * (chi_tau @ u_t)                         # (n, M)

        C00 = norm * (chi_t.t() @ chi_t)                          # (M, M)
        C11 = norm * (gamma.t() @ gamma)                          # (M, M)
        C01 = norm * (chi_t.t() @ gamma)                          # (M, M)

        return u, v, C00, C11, C01, sigma, mu


class VAMPS(nn.Module):
    """Symmetric-rate (S) module of the reversible VAMP-E layer.

    Builds a symmetric ``S`` with the deep_rev_msm constraint ``S @ v = 1``, the
    Koopman ``K = S @ sigma``, and the VAMP-E matrix used as the training loss.

    Parameters
    ----------
    units : int
        Number of metastable states (M).
    activation : callable, default ``torch.exp``
        Applied to the learnable kernel to keep it positive.
    renorm : bool, default False
        Reference heuristic: rescale ``w1`` by ``1/max|w_norm|`` before forming
        ``w2``. Off by default (matches the reference default).
    eps : float
        Numerical guard threshold for the ``(1 - w_norm) / v`` division.
    """

    def __init__(self, units: int, activation=torch.exp, renorm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.M = units
        self.activation = activation
        self.renorm = renorm
        self.eps = eps
        # Reference init: 0.1 * ones(M, M)
        self._s_kernel = nn.Parameter(0.1 * torch.ones(units, units))

    def forward(self, v: torch.Tensor, C00: torch.Tensor, C11: torch.Tensor,
                C01: torch.Tensor, sigma: torch.Tensor):
        """
        Parameters
        ----------
        v : torch.Tensor, shape (M, 1) or (M,)
        C00, C11, C01, sigma : torch.Tensor, shape (M, M)
            As produced by :class:`VAMPU`.

        Returns
        -------
        (vamp_e, K, S) : all shape (M, M)
            ``vamp_e`` is the matrix whose trace is the training loss;
            ``K = S @ sigma`` is the reversible Koopman matrix; ``S`` symmetric.
        """
        v = v.reshape(-1)                                         # (M,)

        kernel_w = self.activation(self._s_kernel)                # (M, M)
        w1 = kernel_w + kernel_w.t()                              # symmetric
        w_norm = w1 @ v                                           # (M,)

        if self.renorm:
            w1 = w1 / torch.clamp(w_norm.abs().max(), min=self.eps)
            w_norm = w1 @ v

        v_safe = _clamp_away_from_zero(v, self.eps, "(1 - w_norm) / v division")
        w2 = (1.0 - w_norm) / v_safe                              # (M,)
        S = w1 + torch.diag(w2)                                   # symmetric, S @ v = 1

        K = S @ sigma
        vamp_e = S.t() @ C00 @ S @ C11 - 2.0 * S.t() @ C01
        return vamp_e, K, S


def vampe_trace_loss(vamp_e: torch.Tensor) -> torch.Tensor:
    """RevGraphVAMP VAMPCE training loss: ``trace(vamp_e)`` (minimize).

    Minimizing ``trace(Sᵀ C00 S C11 − 2 Sᵀ C01)`` maximizes the reversible-K
    VAMP-E score. The constant ``+1`` (constant singular function) is dropped for
    training (irrelevant to gradients); add it back only for reporting via the
    standard VAMP-E score path.
    """
    return torch.trace(vamp_e)


def reversible_vampe_score(vamp_e: torch.Tensor) -> torch.Tensor:
    """Reversible VAMP-E score for logging (higher is better), sans the ``+1``.

    Equals ``-trace(vamp_e)``. For the paper-comparable VAMP-E value (which adds
    the ``+1`` constant), use ``VAMPScore(method='VAMPE')`` on the model outputs.
    """
    return -torch.trace(vamp_e)


# --- Three-phase training schedule (RevGraphVAMP) --------------------------
# Phase 1 ('chi')  : train χ (encoder+classifier) with VAMP-2, χ only.
# Phase 2 ('us')   : freeze χ, train VAMPU+VAMPS with the VAMP-E-trace loss.
# Phase 3 ('all')  : train everything with the VAMP-E-trace loss.
PHASE_CONFIG = {
    'chi': {'train_chi': True,  'train_rev': False, 'loss': 'vamp2'},
    'us':  {'train_chi': False, 'train_rev': True,  'loss': 'vampe'},
    'all': {'train_chi': True,  'train_rev': True,  'loss': 'vampe'},
}


def apply_phase_freeze(chi_params, rev_params, phase):
    """Set ``requires_grad`` per phase and return the params to optimize.

    Pure w.r.t. the phase schedule so it can be unit-tested without the GNN.

    Parameters
    ----------
    chi_params, rev_params : iterable of nn.Parameter
        The χ (encoder+classifier[+embedding]) and reversible (VAMPU+VAMPS) params.
    phase : {'chi', 'us', 'all'}

    Returns
    -------
    (trainable, loss_kind) : (list[nn.Parameter], str)
        Params to hand to the optimizer, and which loss this phase uses
        ('vamp2' or 'vampe').
    """
    if phase not in PHASE_CONFIG:
        raise ValueError(f"unknown phase {phase!r}, expected one of {list(PHASE_CONFIG)}")
    cfg = PHASE_CONFIG[phase]
    chi_params = list(chi_params)
    rev_params = list(rev_params)
    for p in chi_params:
        p.requires_grad_(cfg['train_chi'])
    for p in rev_params:
        p.requires_grad_(cfg['train_rev'])
    trainable = (chi_params if cfg['train_chi'] else []) + \
                (rev_params if cfg['train_rev'] else [])
    return trainable, cfg['loss']
