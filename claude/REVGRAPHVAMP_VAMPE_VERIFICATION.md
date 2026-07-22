# RevGraphVAMP VAMP-E / reversible-layer verification

Verified PyGVAMP's VAMP-E and reversible machinery against the **authoritative
reference implementation**: RevGraphVAMP (Huang et al. 2024),
`github.com/DS00HY/RevGraphVamp`, file `src/revvamp.py`. Purpose: make our
reproduction directly comparable to their reported Table-2 numbers (alanine
VAMP-2 4.41 / VAMP-E 4.38; Aβ42 3.99 / 3.99), and flag issues in their code.

Their reversible layer is a port of Mardt & Noé's **deep reversible MSM**
(`github.com/markovmodel/deep_rev_msm`) — that is the upstream authority.

## 1. VAMP-E score — MATCHES PyGVAMP ✓

Their VAMP-E (in `vamp_score`, method `'VAMPE'`), verbatim:

```python
c00, c0t, ctt = covariances(data, data_lagged, remove_mean=True)
c00_sqrt_inv = sym_inverse(c00, epsilon=epsilon, return_sqrt=True, mode=mode)
ctt_sqrt_inv = sym_inverse(ctt, epsilon=epsilon, return_sqrt=True, mode=mode)
koopman = multi_dot([c00_sqrt_inv, c0t, ctt_sqrt_inv]).t()
u, s, v = torch.svd(koopman)
mask = s > epsilon
u = torch.mm(c00_sqrt_inv, u[:, mask]); v = torch.mm(ctt_sqrt_inv, v[:, mask]); s = s[mask]
u_t = u.t(); v_t = v.t(); s = torch.diag(s)
out = torch.trace(2.*multi_dot([s,u_t,c0t,v]) - multi_dot([s,u_t,c00,u,s,v_t,ctt,v]))
# vamp_score returns:  1 + out     ← constant +1 IS added
```

This is **line-for-line identical** to PyGVAMP's `VAMPScore(method='VAMPE')`
(`pygv/scores/vamp_score_v0.py:89-116`), including the `1 + out` convention.
**So our VAMP-E score is directly comparable to theirs — no change needed.**
(Both add the +1 constant-singular-function term; their reported VAMP-E values
include it.)

## 2. The reproduction target is NOT our current reversible score

RevGraphVAMP does NOT train by minimizing an NLL. It uses two dedicated modules
that build the reversible Koopman from a stationary vector `u` and symmetric `S`,
then **maximizes the VAMP-E score of that reversible K** via a `trace` loss.

**VAMPU** (`_u_kernel`): builds `u = kernel_u / Σ(chi_mean·kernel_u)` and the
reweighted covariances `C00, C11, C01, sigma` (Koopman reweighting).

**VAMPS** (`_s_kernel`), verbatim core:
```python
kernel_w = activation(self._s_kernel)
w1 = kernel_w + kernel_w.t()              # symmetric
w_norm = w1 @ v
w2 = (1 - torch.squeeze(w_norm)) / torch.squeeze(v)
S  = w1 + torch.diag(w2)
K  = S @ sigma
vamp_e = S.t() @ C00 @ S @ C11 - 2 * S.t() @ C01     # the VAMP-E matrix
```

**Loss** = `VAMPCE`: `vamp_score` returns `-trace(vamp_e)`, and
`vampnet_loss = -1 * vamp_score = +trace(vamp_e)` is minimized. Minimizing
`trace(Sᵀ C00 S C11 − 2 Sᵀ C01)` = **maximizing the VAMP-E score of the
reversible K** (the +1 constant is dropped for training; irrelevant to gradients).

**PyGVAMP gap:** our `ReversibleVAMPScore` (`pygv/scores/reversible_score.py`)
uses a *different* parametrization — `K_ij = softplus(S)_ij · u_j / Σ` with
`u = softmax(log_stationary)` — and minimizes **NLL**, not the VAMP-E trace.
**It is a different method and will not reproduce their numbers.** A faithful
reproduction requires porting VAMPU + VAMPS + the VAMP-E-trace (VAMPCE) loss.

## 3. Their three-phase schedule (from `train_ala.py` / `train_ab.py`)

1. Train χ (GCN encoder + softmax classifier) with **VAMP-2**, lr 5e-4 — VAMPU/VAMPS not yet attached.
2. Freeze χ; train **VAMPU+VAMPS** (`u`,`S` kernels) with the **VAMP-E-trace** loss, lr 5e-4.
3. Unfreeze; train **all** with the VAMP-E-trace loss, lr 1e-4.
   (Aβ42 GitHub defaults: pre_train 300 / total 1000 epochs.)

## 4. Observations / issues in their code (documented per request)

None are fatal math errors — the construction is the established deep_rev_msm
one — but several spots are **numerically fragile**, which matters because our
*standard* runs already showed collapse/instability, and this path is more
delicate:

1. **`w2 = (1 - w_norm) / v` (VAMPS)** — elementwise divide by `v = corr_tau @ uᵀ`.
   If any `v` entry is near zero (or sign-flips), `w2` blows up / destabilizes S.
   This is the single most fragile line; a `renorm` heuristic
   (`w1 /= max|w_norm|`) partially guards it but is a band-aid, not a fix.
2. **u-normalization denominator `Σ(chi_mean·kernel_u)`** can approach zero if the
   `_u_kernel` activation lets terms cancel → `u` explodes. Reweighted covariances
   inherit the blow-up. No guard.
3. **`torch.svd`** (VAMP-E path) is **deprecated** — should be `torch.linalg.svd`
   (different return convention: V vs Vᴴ). Not a bug today, but a portability trap.
   NB PyGVAMP's `vamp_score_v0.py` uses the deprecated `torch.svd` too.
4. **`u_kernel = np.log(np.abs(C0inv @ pi))`** init discards sign via `abs`.
   Standard in deep_rev_msm, but a silent assumption of positivity.
5. Cosmetic: `axis=` kwargs (torch prefers `dim=`); no effect.

## 5. Consequence for the reproduction plan

- **Keep** PyGVAMP's VAMP-E score as-is (verified identical) — use it for reporting.
- **Port** VAMPU + VAMPS + VAMP-E-trace (VAMPCE) loss as the reversible training
  path (new modules; do NOT reuse the existing NLL `ReversibleVAMPScore` for the
  reproduction — leave it for backward compat / label it "single-phase NLL").
- **Add** the 3-phase driver (freeze/unfreeze + per-phase loss/lr) in `rev_vampnet.fit()`.
- **Guard** the two fragile lines above (§4.1, §4.2) with epsilon clamps and log
  the guard activations, so we can tell a real collapse from a numerical one.
- Test-first: gradient-isolation per phase; K row-stochastic + detailed balance;
  reversible-VAMP-E trace matches a hand-computed small case.

Reference files: `src/{revvamp.py,utils_vamp.py,train_ala.py,train_ab.py,args.py}`
at `github.com/DS00HY/RevGraphVamp`; upstream `markovmodel/deep_rev_msm`.
