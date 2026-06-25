# Test plan: is the ~0.14 VAMP-2 gap a train/val split-leakage artifact?

> **OUTCOME (executed — hypothesis NOT supported).** Our pipeline already uses the
> random (leaky) split and scores ~4.67 with a real train/val gap. Test 1 (frozen
> re-eval) found blocked ≥ random (concat: random 4.670, blocked 4.756) — the
> *opposite* of leakage. A follow-up batch-size sweep also ruled out per-batch
> estimator bias (perbatch never reaches 4.79; it falls at small batch). The
> blocked-split capability was implemented anyway (`--split_mode blocked`,
> `pygv/dataset/splits.py`). Full results recorded in
> `experiments/trpcage_encoders.md`. Test 2 (paired retrain) was not run — Test 1
> already argues strongly against split leakage. Kept here as the original hypothesis.

*Instructions for an agent (e.g. Claude Code) to implement in the existing GraphVAMPNet
codebase. Self-contained: no external context required beyond the repo.*

## Goal

Our SchNet VAMPNet on Trp-cage scores **VAMP-2 ≈ 4.65** (k=5, τ=20 ns, 10 seeds),
while the reference (Ghorbani et al. 2022, GraphVAMPNet) reports **4.79 ± 0.01** for a
faithfully reproduced architecture. The gap is **flat across encoders** (SchNet ≈ GIN ≈ ML3
all land ~0.14 low), so it is **not** an architecture or expressiveness effect — it lives in
the shared harness.

**Hypothesis to test:** the gap is a **validation-set leakage artifact** caused by a *random
(interleaved)* train/val split of a temporally autocorrelated MD trajectory. The reference
pipeline is built on deeptime's VAMPNet, whose template splits with
`torch.utils.data.random_split` over the time-lagged pair dataset. With 0.2 ns frame spacing,
a validation pair `(x_t, x_{t+τ})` is surrounded by training pairs that are near-identical
configurations, so the learned singular functions are "validated" on effectively memorized
data and the validation VAMP-2 collapses onto the training score (≈ the k=5 ceiling, tiny
std — exactly the signature of 4.79 ± 0.01). A **temporally blocked** split removes that
leakage and should yield the honest, lower score.

If confirmed, the conclusion for the paper is **"4.79 is the leaky-split number; ~4.65 is the
honest one,"** not "we underperform." The leakage offset is encoder-independent and cancels in
any cross-encoder comparison.

## Decision criteria (read first)

Run the tests below and compare full-validation VAMP-2 under the two split types.

| Result | Interpretation | Action |
|---|---|---|
| random ≈ 4.79 **and** blocked ≈ 4.65 | Gap **is** split leakage. Confirmed. | Report both, framed as benchmark correction. |
| random ≈ 4.65 **and** blocked ≈ 4.65 | Split is **not** the cause. | Rule it out cleanly; investigate score estimator / featurization next. |
| random ≈ blocked ≈ 4.79 | You were already on a leaky-equivalent split; something else differs. | Re-check what your current split actually does. |

The key comparison is **paired**: same seeds, same hyperparameters, same scoring function —
only the partition changes.

---

## Test 1 — Freeze-and-re-evaluate (cheap pre-check, ~minutes, no retraining)

Isolates the *evaluation-set* contribution to the inflation. Take an already-trained SchNet
checkpoint (the seed that scored ~4.65) and, **without retraining**, recompute the
full-validation VAMP-2 under different partitions.

1. Load the frozen SchNet lobe/checkpoint. Put it in `eval()` mode; no gradients.
2. Build the full time-lagged dataset over the whole trajectory (same τ in frames as training:
   τ = 20 ns / 0.2 ns = **100 frames**).
3. Compute VAMP-2 (full set, single batch) on each of:
   - **(a) random/interleaved 30%** validation indices (seeded) — mimics the deeptime default.
   - **(b) contiguous blocked 30%** validation indices with a τ-frame seam buffer (see below).
   - (c, optional) the entire trajectory, as a reference ceiling.
4. **Expected if leakage is real:** (a) ≫ (b), with (a) approaching ~4.79 and (b) near ~4.65.

> Caveat: the checkpoint was trained under one specific split, so Test 1 only varies the eval
> partition. It is a fast signal, not the definitive result. If (a) ≈ (b), proceed to Test 2
> anyway — the training-set composition may also matter.

---

## Test 2 — Retrain under both splits (definitive, for the paper)

For each of the **same 10 seeds**, train SchNet from scratch twice — once per split type —
holding *everything else identical* (architecture, epochs=100, lr=5e-4, batch=1000, RBF
dmin=0/dmax=8/step=0.5, k=7 neighbours, residual on). Record the **full-validation** VAMP-2.

```
for seed in SEEDS:                       # the existing 10 seeds
    for split in ["random", "blocked"]:
        set_all_seeds(seed)
        train_idx, val_idx = make_split(split, n_frames, lag=100, val_frac=0.3, seed=seed)
        model = train_schnet(train_idx, ...)         # unchanged training code
        score = full_val_vamp2(model, val_idx)       # see "Scoring" below
        log(seed, split, score)
report_mean_std_per_split()
```

Report `mean ± std` over the 10 seeds for each split. Apply the decision table above.

---

## Split construction (get the seam buffer right)

Time-lagged dataset over `N` frames at lag `τ` (frames) has valid pair-start indices
`t = 0 … N-τ-1`. The split must be defined over these **start indices**.

```python
import numpy as np

def make_random_split(n_pairs, val_frac=0.3, seed=0):
    """Interleaved split — reproduces deeptime's random_split (the leaky baseline)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_pairs)
    n_val = int(n_pairs * val_frac)
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])   # train_idx, val_idx

def make_blocked_split(n_frames, lag, val_frac=0.3, n_blocks=10, seed=0):
    """Temporally blocked split — honest. Whole contiguous blocks go to train or val,
    val blocks spread across the trajectory for representativeness, and any pair whose
    [t, t+lag] window crosses a train<->val seam is DROPPED (a lag-width buffer)."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    block_of = np.zeros(n_frames, dtype=int)
    for b in range(n_blocks):
        block_of[edges[b]:edges[b+1]] = b
    n_val_blocks = max(1, round(n_blocks * val_frac))
    val_blocks = set(rng.choice(n_blocks, size=n_val_blocks, replace=False).tolist())
    train_idx, val_idx = [], []
    for t in range(n_frames - lag):
        in_val_start = block_of[t]       in val_blocks
        in_val_end   = block_of[t + lag] in val_blocks
        if in_val_start != in_val_end:
            continue                     # straddles a seam -> buffer, drop it
        (val_idx if in_val_start else train_idx).append(t)
    return np.array(train_idx), np.array(val_idx)
```

Then wrap with `torch.utils.data.Subset(full_dataset, idx.tolist())`.

**Why blocked-and-spread, not a single tail block:** a single contiguous last-30% holdout may
under- or over-represent metastable states depending on *when* folding events occur in the
trajectory, so a low score there could reflect coverage rather than honest generalization.
Spreading several contiguous blocks across the trajectory (with seam buffers) keeps the
estimate both leakage-free and representative. For extra robustness, average a **blocked k-fold**
(e.g. 5 folds of contiguous chunks) instead of a single draw.

---

## Scoring (avoid convention mismatches)

- **Reuse the exact VAMP-2 scoring function your training loop already calls** to log
  validation score. Do **not** re-implement it for the test — the only thing that should vary
  between conditions is the partition. Re-implementation risks a different `+1`
  (trivial-singular-value) convention or a different regularization `epsilon`, which would
  itself move the number by O(0.1) and confound the test.
- **Evaluate on the full validation set as a single batch** (`batch_size=len(val_ds),
  shuffle=False`), matching deeptime — *not* a mini-batch sampled every N steps. (Our current
  harness logs a noisy quick-validation sampled mid-training; that is a different estimator and
  is a second, smaller reason absolute numbers won't match.)
- Keep `epsilon` / regularization identical to training and identical across both splits.
- Sanity: the VAMP-2 ceiling for k=5 is **5.0** (sum of 5 squared singular values, the first
  being the trivial stationary one = 1). Confirm your scorer's convention puts the ceiling at
  k, so 4.79 / 4.65 are comparable to the reference.

Reference hand-rolled VAMP-2 for cross-checking only (reconcile against your training scorer
before trusting it; mind the trivial-singular-value term):

```python
import torch

def vamp2_reference(chi_t, chi_tau, epsilon=1e-6):
    """chi_t, chi_tau: (n, k) shared-lobe outputs (state probabilities) for x_t and x_{t+τ}."""
    n = chi_t.shape[0]
    chi_t   = chi_t   - chi_t.mean(0, keepdim=True)
    chi_tau = chi_tau - chi_tau.mean(0, keepdim=True)
    c00 = chi_t.T   @ chi_t   / (n - 1)
    ctt = chi_tau.T @ chi_tau / (n - 1)
    c0t = chi_t.T   @ chi_tau / (n - 1)
    def inv_sqrt(C):
        w, V = torch.linalg.eigh(C)
        w = torch.clamp(w, min=epsilon)
        return V @ torch.diag(w.rsqrt()) @ V.T
    K = inv_sqrt(c00) @ c0t @ inv_sqrt(ctt)
    return float(torch.linalg.matrix_norm(K, ord='fro')**2)  # ceiling ~ k for probability lobes
```

---

## What to log

Per run: `seed`, `split_type`, `n_train_pairs`, `n_val_pairs`, `train_vamp2`, `val_vamp2`
(full set), and the per-state singular values if cheap. Persist a clean per-epoch full-val
curve for at least one seed per split (we currently only have noisy sampled points; this is
worth fixing so "underfit vs overfit" rests on real curves).

Tabulate:

```
split      seeds  VAMP-2 (mean ± std)   train_vamp2 (mean)
random       10   ...                   ...      # expect ≈ 4.79, val≈train (leak signature)
blocked      10   ...                   ...      # expect ≈ 4.65, val < train (honest gap)
```

A genuine train→val gap under `blocked` (train near 5.0, val ~4.65) is itself corroborating
evidence; the reference's val sitting at 4.79 ± 0.01 with no such gap is the leak signature.

---

## Optional but recommended: substantive comparison under the honest split

VAMP-2 is saturated here (≈4.65 of a 5.0 ceiling; the SchNet–GIN difference is a fifth of one
seed's std), so it cannot resolve encoders even if the gap is explained. Under the **blocked**
split, also compute the comparisons that have headroom and are the real deliverable:

- **Implied timescales (ITS)** vs lag, and their convergence.
- **Chapman–Kolmogorov test** on the resulting MSM at τ = 20 ns out to ~200 ns.

If SchNet ≈ GIN > ML3 holds on ITS/CK too, that is the result, and the split correction becomes
a calibration footnote rather than a blocker.

---

## Pitfalls checklist

- [ ] Val loader uses **one full batch**, `shuffle=False`. No mid-training sampled scores in the comparison.
- [ ] Blocked split **drops seam-straddling pairs** (τ-width buffer). Otherwise it still leaks at block boundaries.
- [ ] **Same scoring function and `epsilon`** for both splits; reuse the training scorer.
- [ ] **Same 10 seeds** across both split conditions (paired comparison).
- [ ] `random_split` is **seeded** (deeptime's default uses the global RNG — pin it).
- [ ] Confirm the VAMP-2 ceiling convention is **k = 5** so numbers are comparable to 4.79.
- [ ] Val blocks collectively **span the major metastable states** (spread them; or use blocked k-fold).
- [ ] Lag is **100 frames** (20 ns / 0.2 ns), consistent with the reference and across both splits.
