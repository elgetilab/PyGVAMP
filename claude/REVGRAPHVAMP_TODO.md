# RevGraphVAMP reproduction — TODO / scoping (grounding truth)

Single source of truth for the RevGraphVAMP (reversible) reproduction. Written
2026-07-22 because the originating session got long; a fresh context should be
able to continue from this file alone. Companion: `REVGRAPHVAMP_VAMPE_VERIFICATION.md`
(the code verification) and `EXPERIMENT_CHECKLIST.md` (Category 1, RevGraphVAMP rows).

---
## ⭐ RESTART / RESUME (written 2026-07-23, before a planned cluster restart)

**State at restart:** working tree CLEAN; no running/queued SLURM jobs (nothing
for the restart to kill). All work committed. HEAD = `0a4a66f`.

**⚠️ BEFORE the restart — PUSH (I could not; do it from your env):**
`git push origin main`, then confirm `origin/main` is at `0a4a66f`. Local
`origin/main..HEAD` reads empty here but my shell can't reach GitHub to verify,
so confirm the true remote yourself. This is the #1 thing to secure — the ~8
RevGraphVAMP commits (809e6f3 → 0a4a66f) live only on this disk otherwise.

**Survives the restart (on-disk, if the disk persists):** the repo + all commits;
data at `/mnt/hdd/data/alanine/` (750k frames) and `/mnt/hdd/data/ab42/trajectories/red/`
(5119 xtc); the deployed module + conda env at `/opt/software/pygvamp/1.0.0`.

**Module note:** the DEPLOYED module LACKS the reversible-3-phase code, so all
RevGraphVAMP runs must use the working-tree code via
`--export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP` (the alanine
& ab42 scripts document this). To bake it into the module instead, rebuild:
`sudo bash module/install_module.sh --prefix /opt/software/pygvamp/1.0.0 --moduledir /opt/modulefiles --skip-env`.

**RESUME HERE → step 5c (GPU smoke on REAL unstrided alanine, then full runs).**
First, a modest real-data GPU smoke (one seed, few epochs) to confirm a
NON-degenerate VAMP-2 climbing toward the ~4.41 regime (the earlier CPU smoke used
heavy stride + 2–3 epochs → numbers meaningless, only the code path was checked):
```
sbatch --array=0 --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
    cluster_scripts/alanine_rev_v1_array.sh      # then read val VAMP-2 in the .out
```
(or edit EPOCH_CHI/EPOCH_ALL down to ~20/30 for a quicker smoke first). If VAMP-2
looks sane, launch the full 10-seed arrays (alanine then Aβ42 — time one Aβ42 seed
first, it's ~1.26M frames × 1300 epochs). Aggregate with
`cluster_scripts/aggregate_reversible_array.py` (targets: alanine 4.41/4.38,
Aβ42 3.99/3.99). Still-open before trusting numbers: confirm alanine's exact
pre_train/epochs from their run cmd (Aβ42 uses GitHub 300/1000).

## Implementation summary (what is built — all committed, all tested)
- `pygv/scores/reversible_vampe.py` — VAMPU/VAMPS reversible layer, `vampe_trace_loss`,
  phase-freeze (`PHASE_CONFIG`/`apply_phase_freeze`), and the algebraic init
  (`matrix_inverse`, `covariances_E`, `compute_pi`, `algebraic_init_us`).
- `pygv/vampnet/rev_vampnet.py` — `attach_vampe_layer`, `fit_three_phase`
  (χ VAMP-2 → algebraic U/S init → joint VAMP-E; best-tracked; writes best_model.pt).
  Old single-phase NLL `fit()` kept for back-compat.
- `pygv/pipe/{args.py,training.py}` + `pygv/config/base_config.py` — CLI flags
  (`--rev_three_phase`, `--epoch_chi/us/all`, `--lr_chi/us/all`, `--rev_activation`,
  `--rev_renorm`) plumbed through config → `_train_reversible_three_phase`.
- `cluster_scripts/{alanine_rev_v1_array.sh, ab42_rev_v1_array.sh,
  aggregate_reversible_array.py, download_alanine.sh}`; tracker
  `experiments/revgraphvamp_repro.md`.
- Tests: `tests/test_reversible_vampe.py` (16 pass). VAMP-E verified identical to
  their repo; full 3-phase smoke passed end-to-end (CPU). Full suite unaffected
  (only pre-existing torch_scatter/Meta failures, unrelated).
---

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

## Data status — RESOLVED 2026-07-22 (step 4 done)

- **Alanine dipeptide: DOWNLOADED** to `/mnt/hdd/data/alanine/` via
  `cluster_scripts/download_alanine.sh` (mdshare mirror; mdshare pkg not installed
  in the env). 3× `alanine-dipeptide-N-250ns-nowater.xtc` + `...-nowater.pdb`.
  Verified: 22 atoms / **10 heavy** (ACE-ALA-NME), **750,000 frames** (3×250k) —
  matches the RevGraphVAMP spec exactly. Selection: `not element H` (= 10 heavy).
  Timestep 0.001 ns (1 ps/frame); lag 20 ps = `--lag_times 0.02 --timestep 0.001`.
- **Aβ42: already on disk, no combining needed — CORRECTION.** The earlier
  "reproduction needs red+ox combined" assumption was WRONG. RevGraphVAMP's Aβ42 is
  the **reduced (red) ensemble ALONE**: `/mnt/hdd/data/ab42/trajectories/red/` has
  **exactly 5119 xtc** files — matching the paper's stated "5,119 trajectories" —
  and RevGraphVAMP's own repo stores its data under `trajectories/red/`. The `ox`
  ensemble (3071 trajs) is a separate system they did NOT report; leave it out.
  Feed with `--traj_dir .../ab42/trajectories/red/ --file_pattern '*.xtc'`
  (recursive glob is ON by default → picks up the nested rN/rNcs subdirs).
  Topology `.../red/topol.pdb`: 42 residues, **42 CA** → use `name CA` (42),
  resolving the paper(40)-vs-GitHub(42) discrepancy in favor of 42.

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
- [x] (4) Data prep — DONE 2026-07-22. Alanine downloaded+verified (750k frames,
      10 heavy) via `download_alanine.sh`; Aβ42 = red ensemble already on disk
      (5119 trajs, 42 CA), no combining needed (see corrected Data status above).
- [~] (5) SLURM scripts + aggregator + tracking; run 10 seeds each ← IN PROGRESS
      DONE so far: alanine data + alanine_rev_v1_array.sh; pipeline config plumbing
      fixed (base_config fields + args→config map, commit c65a574); smoke run
      PASSED end-to-end (chi/us/all dispatch, U+S freeze = 2 params, best_model,
      analysis, PIPELINE COMPLETED). Schedule-fidelity resolved: their protocol =
      χ-VAMP2 → algebraic U/S init → joint VAMP-E; decision = implement faithfully.
      REMAINING: (5a) algebraic init + driver rewire + test [spec above];
      (5b) Aβ42 script + reversible aggregator + tracking; (5c) GPU smoke then
      10-seed runs (alanine + Aβ42). Recommend doing 5a as a fresh focused step.

## SCHEDULE FIDELITY — RESOLVED against their train_ab.py (2026-07-23)

Their ACTUAL protocol is NOT 3 gradient phases. From `train_ab.py`:
1. Stage 1 (`pre_train_epoch`): freeze VAMPU/VAMPS, train χ with **VAMP-2**.
2. Stage 2 (**algebraic, no gradient**): transform all data through frozen χ →
   `vampnet.update_auxiliary_weights([probs, probs_tau], optimize_S=True)` —
   closed-form init of the u/S kernels from the encoder covariances
   (`_compute_pi` / `optimize_S`).
3. Stage 3 (`epochs`): unfreeze all, joint-train with **VAMPCE** (VAMP-E trace).

OUR driver (`fit_three_phase`) does 3 GRADIENT phases (chi / us / all) — phase
`us` gradient-trains U/S with frozen χ instead of the algebraic init. This is a
**documented deviation**. Faithful reproduction needs Stage-2 algebraic init
(port `update_auxiliary_weights`/`optimize_S`/`_compute_pi` from revvamp.py, add
a correctness test) and then chi→init→all (phase `us` becomes optional/removed).
DECISION PENDING (see chat): (A) implement algebraic init [faithful] vs
(B) keep the gradient-phase-us approximation [documented deviation].

Smoke run (2/2/2, alanine, CPU) PASSED end-to-end after the config-plumbing fix:
dispatch chi(56 params,VAMP-2)/us(2 params,VAMP-E)/all(58,VAMP-E), U+S freeze
verified, best_model.pt saved, analysis + PIPELINE COMPLETED. The mechanism is
sound; only the Stage-2 init fidelity is open.

## Algebraic U/S init — VERBATIM spec (decision: implement faithfully, 2026-07-23)

Port these from `DS00HY/RevGraphVamp/src/revvamp.py` (chi_0/chi_t are the frozen-χ
softmax outputs over the whole train set; assign into VAMPU._u_kernel / VAMPS._s_kernel):

```python
def matrix_inverse(mat, epsilon=1e-10):          # eigh pseudo-inverse
    eigva, eigveca = np.linalg.eigh(mat.detach().cpu().numpy())
    inc = eigva > epsilon
    eigv, eigvec = eigva[inc], eigveca[:, inc]
    return eigvec @ np.diag(1./eigv) @ eigvec.T

def covariances_E(chil, chir):                   # NOT mean-removed
    norm = 1./chil.shape[0]
    C0, Ctau = norm * chil.T @ chil, norm * chil.T @ chir
    return matrix_inverse(C0), Ctau              # (C0inv, Ctau)

def _compute_pi(K):                              # stationary via left-eigvec @ eigval≈1
    eigv, eigvec = np.linalg.eig(K.T)
    pi_v = eigvec[:, ((eigv - 1)**2).argmin()]
    return pi_v / pi_v.sum(keepdims=True)

# update_auxiliary_weights(data=[chi_0, chi_t], optimize_u=True, optimize_S=True):
C0inv, Ctau = covariances_E(chi_0, chi_t)
K = C0inv @ Ctau                                  # non-reversible Koopman
# optimize_u:
pi = _compute_pi(K);  u_kernel = np.log(np.abs(C0inv @ pi))   # -> VAMPU._u_kernel
# optimize_S (after a vampu forward to get sigma):
sigma_inv = matrix_inverse(sigma)
S_nonrev = K @ sigma_inv
S_rev = 0.5*(S_nonrev + S_nonrev.T)
s_kernel = np.log(np.abs(0.5 * S_rev))            # -> VAMPS._s_kernel
```

WHY exp activation is right: forward does `activation(exp)(_u_kernel)` → `exp(log|·|)=|·|`,
so the log-init inverts the exp activation. (Guard the `log(abs(...))` against zeros.)

### Revised step-5 plan
- [x] (5a) DONE 2026-07-23. `reversible_vampe.py`: `matrix_inverse`, `covariances_E`,
  `compute_pi`, `algebraic_init_us` (faithful port). `rev_vampnet.py`:
  `_collect_chi` + `fit_three_phase` rewired to chi(VAMP-2) → algebraic U/S init →
  all(VAMP-E), with `algebraic_us_init=True` default (gradient `us` kept as opt-in
  fallback). Best model tracked after init AND each phase-3 epoch. Tests (16 pass):
  matrix_inverse vs np.inv, compute_pi vs known stationary, covariances_E vs manual
  Koopman, and metastable-data init recovers reversible VAMP-E to near the standard
  ceiling (−1.76 → 1.81 vs 1.94). Smoke: chi → "algebraic U/S init done" → all →
  PIPELINE COMPLETED. `epoch_us` now ignored in faithful mode (alanine script updated).
- [x] (5b) DONE 2026-07-23. `cluster_scripts/ab42_rev_v1_array.sh` (red dir, 42 CA,
  k4, lag10ns@0.25, batch500, Gaussians [0,8], epoch_chi300/epoch_all1000, src
  override); `cluster_scripts/aggregate_reversible_array.py` (parses
  `val VAMP-2=/VAMP-E=`, reports cross-seed at max-VAMP-2 epoch; regex validated);
  tracker `experiments/revgraphvamp_repro.md`. Alanine script schedule updated
  (epoch_us ignored). Args validated (--distance_min/max exist), both scripts bash -n clean.
- (5c) Alanine smoke on GPU with real (unstrided) data + a few epochs to sanity a
  non-degenerate VAMP-2, THEN 10-seed runs for alanine + Aβ42.

## Open questions to resolve before RUNNING (not before coding)

- Exact `self.activation` for the u/S kernels (fetch `__init__`).
- Exact `epoch_chi/epoch_us/epoch_all` for alanine (Aβ42: 300 pre / 1000 total).
- Aβ42 atom count 40 vs 42 — run both, see which matches 3.99.
- Whether their χ (phase-1 VAMP-2) uses the same GCN/SchNet encoder we do, or GASchNet.
