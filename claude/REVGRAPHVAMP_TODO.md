# RevGraphVAMP reproduction — TODO / scoping (grounding truth)

Single source of truth for the RevGraphVAMP (reversible) reproduction. Written
2026-07-22 because the originating session got long; a fresh context should be
able to continue from this file alone. Companion: `REVGRAPHVAMP_VAMPE_VERIFICATION.md`
(the code verification) and `EXPERIMENT_CHECKLIST.md` (Category 1, RevGraphVAMP rows).

## Goal & success criteria

Reproduce RevGraphVAMP (Huang et al. 2024) Table 2 with PyGVAMP:
- **Alanine dipeptide**: k=6, lag=20 ps, VAMP-2 = **4.41 ± 0.01**, VAMP-E = **4.38 ± 0.01**.
- **Aβ42 (combined)**: k=4, lag=10 ns, VAMP-2 = **3.99 ± 0.002**, VAMP-E = **3.99 ± 0.003**.
Success = 10-seed mean ± CI overlaps their value or Δ < 0.05.

## Decisions locked (this session)

1. Stay on the 5090 cluster (not the 4060 Ti) for these runs.
2. Implement the **3-phase** RevGraphVAMP schedule properly (not single-phase NLL).
3. Implement **VAMP-E properly** — matched to their repo, not derived from paper text.
4. Verified against `github.com/DS00HY/RevGraphVamp` `src/revvamp.py`:
   - **Their VAMP-E score == PyGVAMP's `VAMPScore(method='VAMPE')`** (identical SVD
     formula + `1 + out`). KEEP ours; it is directly comparable. No change.
   - Their reversible **training** uses VAMPU/VAMPS modules + a VAMP-E-trace loss
     (VAMPCE), a port of `markovmodel/deep_rev_msm`. PyGVAMP's existing
     `ReversibleVAMPScore` (softplus-K, NLL) is a DIFFERENT method → cannot
     reproduce their numbers. **Must port VAMPU/VAMPS + VAMPCE.**

## Reference code (verbatim, the port target)

Source: `DS00HY/RevGraphVamp/src/revvamp.py`. Upstream authority:
`markovmodel/deep_rev_msm`.

**VAMPU.forward(x)** — x = (chi_t, chi_tau):
```python
n_batch = chi_t.shape[0]; norm = 1./n_batch
corr_tau = norm * (chi_tau.t() @ chi_tau)
chi_mean = torch.mean(chi_tau, axis=0, keepdims=True)
kernel_u = torch.unsqueeze(self.activation(self._u_kernel), axis=0)
u = kernel_u / torch.sum(chi_mean * kernel_u, 1, keepdims=True)
u_t = u.t()
v = corr_tau @ u_t
mu = norm * (chi_tau @ u_t)
sigma = (chi_tau * mu).t() @ chi_tau
gamma = chi_tau * (chi_tau @ u_t)
C00 = norm * (chi_t.t() @ chi_t)
C11 = norm * (gamma.t() @ gamma)
C01 = norm * (chi_t.t() @ gamma)
return [tile(var,n_batch) for var in (u,v,C00,C11,C01,sigma)] + [mu]
```

**VAMPS.forward(x)** — x = (v,C00,C11,C01,sigma) or (chi_t,chi_tau,u,v,C00,C11,C01,sigma):
```python
kernel_w = self.activation(self._s_kernel)
w1 = kernel_w + kernel_w.t()          # symmetric
w_norm = w1 @ v
if self.renorm: w1 = w1 / torch.max(torch.abs(w_norm)); w_norm = w1 @ v
w2 = (1 - torch.squeeze(w_norm)) / torch.squeeze(v)
S = w1 + torch.diag(w2)
K = S @ sigma
vamp_e = S.t() @ C00 @ S @ C11 - 2 * S.t() @ C01     # VAMP-E matrix
# returns vamp_e (tiled), K, probs, S
```

**Loss (VAMPCE):** `vamp_score(...,method='VAMPCE')` returns `-trace(vamp_e)`;
`vampnet_loss = -1 * vamp_score = +trace(vamp_e)` is MINIMIZED. Minimizing
`trace(Sᵀ C00 S C11 − 2 Sᵀ C01)` == maximizing the reversible-K VAMP-E score
(the +1 constant is dropped for training, irrelevant to gradients).

**STILL NEEDED (fetch when implementing):** the `VAMPU.__init__` / `VAMPS.__init__`
(param shapes, the exact `self.activation` — likely `exp`/`abs`/`softplus`, `M`=n_states,
`renorm` default) and the `_compute_pi` / `optimize_S` init helpers. Get from raw
`revvamp.py`. Also confirm exact `epoch_chi/us/all` + batch/lr from `train_ala.py`
and `train_ab.py` (checklist says Aβ42 pre_train=300/total=1000; alanine TBC).

## Three-phase schedule (target)

1. Train χ (encoder+classifier) with **VAMP-2**, lr 5e-4, `epoch_chi` epochs (VAMPU/VAMPS detached).
2. Freeze χ; train **VAMPU+VAMPS** with **VAMP-E-trace** loss, lr 5e-4, `epoch_us` epochs.
3. Unfreeze; train **all** with VAMP-E-trace, lr 1e-4, `epoch_all` epochs.

Shared hp (both systems): hidden_dim=16, n_graph_layers=4, n_Gaussians=16, batch
1000 (alanine) / 500 (Aβ42), 70/30 split. Alanine: 10 heavy atoms, n_neighbors=5,
750k frames (3×250ns×1ps). Aβ42: 40 vs 42 atoms (paper Table1=40 vs GitHub `--num-atoms 42`
— TEST BOTH), n_neighbors=10, 1.26M frames, 250 ps/frame, dmin0/dmax8/step0.5.

## Data status

- **Alanine dipeptide: NOT on disk.** `/mnt/hdd/data/` has none. Download via
  mdshare (`alanine-dipeptide-*-northo.npz` or the 3 DESRES-style trajs). Small.
- **Aβ42: on disk but SPLIT** at `/mnt/hdd/data/ab42/trajectories/{red,ox}/rN[cs]/*.xtc`,
  no `combined/` dir. Reproduction needs COMBINED (ox+red together; the ox/red
  split is our separate novel contribution, not RevGraphVAMP's protocol). Point at
  the `trajectories/` parent with a recursive glob, or assemble a combined list.
  Topology: `/mnt/hdd/data/ab42/trajectories/red/topol.pdb` (verify ox matches).

## Code fragilities to guard (from their code — see verification doc §4)

- `w2 = (1 - w_norm)/v`: divide by near-zero/sign-flipping `v` → S blows up. Guard
  with an epsilon-clamped denominator; LOG when the guard fires (distinguish real
  collapse from numerical).
- u-normalization `Σ(chi_mean·kernel_u)` can hit ~0 → clamp; log.
- Replace deprecated `torch.svd` → `torch.linalg.svd` in any NEW code (leave
  existing `vamp_score_v0.py` alone unless it bites).

## File-by-file plan

1. **NEW `pygv/scores/reversible_vampe.py`** — `VAMPU`, `VAMPS` modules +
   `vampe_trace_loss` (= `trace(vamp_e)`), faithful to the verbatim above, with the
   §4 guards. (Do NOT touch `reversible_score.py`; keep the NLL path for back-compat,
   relabel as "single-phase NLL".)
2. **`pygv/vampnet/rev_vampnet.py`** — rewrite `fit()` into a 3-phase driver:
   per-phase param groups (freeze/unfreeze via requires_grad), per-phase loss
   (VAMP-2 → VAMP-E-trace) and lr; attach VAMPU/VAMPS; best-model tracking on the
   VAMP-E score during phase 3.
3. **`pygv/args/args_train.py` (+ `pipe/args.py`)** — add `--epoch_chi/--epoch_us/
   --epoch_all`, `--lr_chi/--lr_us_all` (defaults 5e-4/5e-4/1e-4). Gate on `--reversible`.
4. **`pygv/pipe/training.py`** — wire phase args + VAMPU/VAMPS into the reversible path.
5. **`tests/test_reversible_vampe.py`** — TEST FIRST: (a) VAMPS `vamp_e` trace matches
   a hand-computed tiny case; (b) K row-stochastic + detailed balance (u_i K_ij =
   u_j K_ji); (c) phase gradient isolation (phase1 leaves u/S kernels untouched;
   phase2 leaves χ frozen; phase3 updates all); (d) guards fire + log on degenerate v.
6. **Data prep**: mdshare alanine download script; Aβ42-combined assembly.
7. **SLURM scripts** (mirror existing patterns, params in-script): `alanine_rev_*.sh`,
   `ab42_combined_rev_*.sh`. 10 seeds each. + aggregator + tracking `.md`.

## Step ordering & status

- [x] **(1) Port VAMPU/VAMPS + VAMP-E-trace loss + correctness test** — DONE 2026-07-22.
      `pygv/scores/reversible_vampe.py` (VAMPU, VAMPS, vampe_trace_loss,
      reversible_vampe_score, guarded); `tests/test_reversible_vampe.py` (8 tests
      pass, incl. golden `S @ v = 1` for M=2/3/5). Activation defaulted to
      `torch.exp` — CONFIRM vs their train script before a strict run.
- [x] (2) 3-phase driver — DONE 2026-07-22. Added to `rev_vampnet.py` (kept old
      single-phase `fit()` for back-compat): `attach_vampe_layer()`,
      `chi_parameters()`, `reversible_vampe_parameters()`, `_phase_loss()`,
      `_validate_scores()`, `fit_three_phase(epoch_chi/us/all, lr_chi/us/all)`.
      Phase freeze logic (`PHASE_CONFIG`, `apply_phase_freeze`) in
      `reversible_vampe.py`. Best model tracked by val VAMP-2 in phase 3.
      Tests: gradient-isolation verified (χ moves only in chi/all; U+S only in
      us/all). Full suite still collects (658 tests, no import breakage).
- [x] (3) CLI args + pipeline wiring — DONE 2026-07-22.
      `pipe/args.py`: `--rev_three_phase`, `--epoch_chi/us/all`, `--lr_chi/us/all`,
      `--rev_activation {exp,abs,softplus}` (default exp), `--rev_renorm`.
      `pipe/training.py`: `_train_reversible_three_phase()` dispatched from
      `train_model` when `--reversible --rev_three_phase`; calls
      `attach_vampe_layer` → `fit_three_phase`. Fixed `fit_three_phase` to write
      `best_model.pt`/`final_model.pt` so the pipeline discovers the model.
      NOTE: args added to `pipe/args.py` only (the `pygvamp` CLI parser); NOT
      `args/args_train.py` (standalone train entry, unused by the repro). Add
      there too only if a standalone train path is needed.
      Activation defaulted to exp — CONFIRM vs their train_ala.py/train_ab.py.
- [ ] (4) Alanine data (mdshare) + Aβ42-combined prep
- [ ] (5) SLURM scripts + aggregator + tracking md; then run 10 seeds each

## Open questions to resolve before RUNNING (not before coding)

- Exact `self.activation` for the u/S kernels (fetch `__init__`).
- Exact `epoch_chi/epoch_us/epoch_all` for alanine (Aβ42: 300 pre / 1000 total).
- Aβ42 atom count 40 vs 42 — run both, see which matches 3.99.
- Whether their χ (phase-1 VAMP-2) uses the same GCN/SchNet encoder we do, or GASchNet.
