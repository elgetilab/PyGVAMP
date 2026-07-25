# RevGraphVAMP reproduction (reversible / VAMP-E) — tracker

Reproduces RevGraphVAMP (Huang et al. 2024, `github.com/DS00HY/RevGraphVamp`)
Table 2 with PyGVAMP's ported reversible VAMP-E layer + faithful 3-phase schedule.
Implementation + verification details: `claude/REVGRAPHVAMP_TODO.md` and
`claude/REVGRAPHVAMP_VAMPE_VERIFICATION.md`. This file tracks the runs.

## Method (faithful to their train_ab.py)

χ-pretrain with **VAMP-2** (`epoch_chi`) → **algebraic closed-form init** of the
VAMPU/VAMPS (u, S) kernels from the frozen-χ covariances (no gradient) → joint
**VAMP-E** training (`epoch_all`). Reversible layer = ported VAMPU/VAMPS
(`pygv/scores/reversible_vampe.py`); driver = `RevVAMPNet.fit_three_phase`
(`--reversible --rev_three_phase`). VAMP-E score verified identical to theirs.

Reported per seed = (VAMP-2, VAMP-E) at the epoch of max val VAMP-2 (model
selection). Cross-seed via `cluster_scripts/aggregate_reversible_array.py`.

## Systems & targets (Table 2)

| System | k | lag | atoms / selection | target VAMP-2 | target VAMP-E |
|---|---|---|---|---|---|
| Alanine dipeptide | 6 | 20 ps | 10 heavy (`not element H`) | 4.41 ± 0.01 | 4.38 ± 0.01 |
| Aβ42 (reduced) | 4 | 10 ns | 42 CA (`name CA`) | 3.99 ± 0.002 | 3.99 ± 0.003 |

- **Alanine data**: `/mnt/hdd/data/alanine/` (3×250 ns nowater, 750k frames, 1 ps/frame;
  `download_alanine.sh`). timestep 0.001, lag 0.02 ns, n_neighbors 5.
- **Aβ42 data**: `/mnt/hdd/data/ab42/trajectories/red/` (the reduced ensemble =
  RevGraphVAMP's Aβ42; 5119 xtc, ~1.26M frames, 250 ps/frame). timestep 0.25,
  lag 10 ns, n_neighbors 10, Gaussians over [0,8]. NOT combined with ox.

Shared: SchNet χ, hidden 16, output 16, 4 interactions, 16 Gaussians, no embedding,
clf 1-layer/no-norm, 70/30 split, 10 seeds. Epochs (their GitHub): Aβ42
pre_train 300 / epochs 1000; alanine provisional 100/100 (confirm from their run cmd).

## Scripts

| Script | System |
|---|---|
| `cluster_scripts/alanine_rev_v1_array.sh` | Alanine, 10-seed array |
| `cluster_scripts/ab42_rev_v1_array.sh` | Aβ42 (red), 10-seed array |
| `cluster_scripts/aggregate_reversible_array.py` | cross-seed VAMP-2/VAMP-E |

Both need the working-tree reversible-3-phase code until the module is rebuilt →
submit with `--export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP`.

## Status

| System | SchNet-χ / rev impl | GPU smoke | 10-seed run | result |
|---|---|---|---|---|
| Alanine | ✓ implemented; pipeline validated end-to-end on real data | ✓ ran, but **χ VAMP-2 ceilings at ~2.85** (target 4.41) | ☐ BLOCKED | χ-stage ceiling |
| Aβ42 | ✓ script ready | ☐ (blocked pending alanine diagnosis) | ☐ | — |

## ✅ ALANINE REPRODUCED (10-seed, 2026-07-25)

Full 4-stage schedule at tau=1 (lag 0.001 ns), encoder v2, h_g=8, k=6, 5 nbrs,
batch 1000. Model-selected VAMP scores, `alanine_rev_v1` seeds 0–9:

  **VAMP-2 = 4.402 ± 0.244** (paper 4.41 ± 0.01)  Δ = −0.008
  **VAMP-E = 4.402 ± 0.244** (paper 4.38 ± 0.01)  Δ = +0.022

Mean matches almost exactly. Caveat: our seed variance is ~24× theirs (±0.24 vs
±0.01). 8/10 seeds cluster in 4.46–4.53; two land low (seed 9 = 3.76, seed 4 =
4.22, the partial-collapse failure mode). Their ±0.01 implies no collapsed seeds —
an unmatched training-stability difference (init/anti-collapse detail unknown), NOT
a score gap. Kept all seeds (no cherry-picking). Per-seed: 4.494/4.504/4.458/4.519/
4.218/4.521/4.518/4.503/4.526/3.760.

## ✅ RESOLVED 2026-07-24 — the gap was LAG TIME (τ), not the model

**Their reported alanine 4.41 is at lag = 1 ps, not 20 ps.** `args.py` defaults to
`--tau 1` = ONE FRAME, and their alanine data is 750k frames at 1 ps/frame. We had
assumed 20 ps from the paper prose. Measured on our SchNet-χ (k=6, single seed):

| lag | χ VAMP-2 |
|---|---|
| 20 ps (what we ran all along) | 2.85 |
| 5 ps | 3.51 |
| **1 ps (their `--tau 1`)** | **4.41–4.47** (peak 4.4668; VAMP-E 4.414) |

Target is 4.41 / 4.38 → reproduced at their actual lag. VAMP-2 = 1 + Σσᵢ² with
σᵢ = e^(−τ/tᵢ), so a 20× longer lag decays every singular value; scores are simply
not comparable across lags. **Nothing was wrong with the encoder, the reversible
port, or the algebraic init.** Second time this class of bug bit us (cf. villin
τ-normalization).

Ruled out beforehand, at real GPU cost: encoder variant v1/v2, n_neighbors 5/9,
batch 200/1000, seeds 1–4, χ epochs to 120, joint epochs to 110, and joint lr
∈ {1e-4, 5e-4, 5e-3, 0.2}. All held ~2.85 at 20 ps — the invariance to every
training knob is what finally pointed at the data/scoring convention.

### Paper checked directly (bioRxiv PDF, 2026-07-24) — NOT a paper error, OUR misread

Downloaded `10.1101/2024.03.11.584426v1.full.pdf` and read the text:

- **Table 1 "Hyperparameters of model training" has NO lag/tau column.** The paper
  never states the training lag. Columns are: graph layers, neurons, states,
  batch-size, learning rate, atoms, neighbors, Gaussians.
- The only τ values in the paper are **CK-test / ITS lags**, not training lags:
  alanine *"The implicit time function of the trained model is depicted in Figure 3a,
  with a selected lag time of τ=20 ps"* (Figure 3 = *"Correctness verification …
  (a) Implied timescale (ITS) (b) CK test results"*), and Aβ42 *"the implied
  timescales … converged when the lag time τ=10 ns. Therefore, **for the subsequent
  correctness test (CK test)**, a lag time of τ=10 ns is [used]"*.
- Our `EXPERIMENT_CHECKLIST.md:128` claim "lag time = 20 ps (confirmed in Section
  3.1.1)" **misattributed the CK-test lag as the training lag**. That is the whole
  bug. With no training lag stated, their code default `--tau 1` (= 1 frame = 1 ps)
  governs Table 2's VAMP scores — which is exactly where we reproduce 4.41.

**Table 1 values confirm several of our settings and settle an open question:**

| | layers | neurons | states | batch | lr | atoms | neighbors | Gaussians |
|---|---|---|---|---|---|---|---|---|
| Alanine | 4 | 16 | 6 | **1000** | **[0.0005, 0.0001]** | 10 | **5** | 16 |
| Aβ42 | 4 | 16 | 4 | 500 | [0.0005, 0.0001] | **40** | 10 | 16 |

- **lr = [0.0005, 0.0001]** ⇒ `set_optimizer_lr(0.2)` IS a factor (0.2 × 5e-4 = 1e-4).
  Our `lr_chi 5e-4 / lr_all 1e-4` is confirmed correct by the paper, independently of
  the collapse experiment.
- **batch 1000** and **5 neighbors** for alanine confirm our original settings (the
  bs=200 / nn=9 probes were chasing nothing — those came from Aβ42-oriented arg defaults).
- **Aβ42 = 40 atoms in the paper** vs `--num-atoms 42` in their GitHub command, and
  their `topol.pdb` has 42 CA. The 40-vs-42 question is still OPEN; run both.

## (historical) BLOCKER as diagnosed 2026-07-23 — χ-stage VAMP-2 ceilings at ~2.9

GPU smoke (job 796, real unstrided 750k frames, χ=20/all=30) ran cleanly
end-to-end (correct data 10 heavy atoms / lag 20 ps; chi→algebraic init→joint
VAMP-E → PIPELINE COMPLETED) but **VAMP-2 plateaus at ~2.85 vs the 4.41 target,
with 3 of 6 states collapsed to 0% population** (analysis recommended retrain @3).

Diagnosis (all committed to this tracker + TODO):
1. **Ceiling is in the χ stage, not the reversible port.** The χ phase trains a
   plain VAMP-2 softmax classifier with VAMPU/VAMPS detached (= a standard
   VAMPNet). Added per-epoch χ validation logging to `fit_three_phase`
   (`rev_vampnet.py`); χ VAMP-2 **saturates at ~2.8 by epoch 5** and is flat
   through epoch 120 (job 797). Not under-training.
2. **Encoder architecture matches** — their χ is also SchNet
   (`conv_type default='SchNet'` in their args.py). Not an architecture gap.
3. **VAMP-2 score convention is identical** — both are `1 + ‖whitened_koopman‖²_F`
   with mean removed (deeptime convention). So 2.9 is a REAL deficiency, NOT a
   normalization artifact (unlike the villin τ-normalization case).
4. **Ruled out n_neighbors and batch_size** (χ-only probes, jobs 798/799/800):
   nn=9 → 2.90 (marginal); bs=200 → 2.58 (worse); nn9+bs200 → 2.73 (worse).
   The ~2.8–2.9 ceiling is robust to these knobs.

### ⚠️ RETRACTED (2026-07-24): "they use a graph-attention encoder"

An earlier version of this file claimed Table-2's 4.41 came from a
NeighborMultiHeadAttention encoder and that SchNet could not reach it. **That was
WRONG** — it came from an unreliable web-summary, not their source. Verified by
cloning `github.com/DS00HY/RevGraphVamp` and reading the code directly:

- Their README training command passes **`--conv_type SchNet`** explicitly (plus
  `--residual`). `train_ala.py` does `lobe = GraphVampNet()` and never overrides
  `conv_type`, whose default in `args.py` is `'SchNet'`.
- `GraphVampNet` is a *wrapper* over 4 interchangeable convs (GraphConvLayer,
  NeighborMultiHeadAttention, GATLayer, SchNet/InteractionBlock). SchNet is a
  first-class choice, NOT a weak baseline.
- **Our SchNet encoder choice is correct.** No attention encoder port is needed.

### Verified against their source (2026-07-24, local clone)

- **Alanine = 5 neighbors** (`--data-path` default `../intermediate/ala_5nbrs_1ns_`).
  Our original `n_neighbors=5` was right; the nn=9 probe was chasing nothing.
- **Alanine = 750,000 frames** (`ala_5nbrs_1ns_datainfo.npy` → 3×250,000).
  Our dataset matches theirs exactly.
- Aβ42 command (README): `--num-atoms 42 --num-classes 4 --num_neighbors 10
  --conv_type SchNet --dmin 0 --dmax 8. --step 0.5 --batch-size 500 --lr 0.0005
  --pre-train-epoch 300 --epochs 1000 --residual --score-method VAMPCE`.
- **Their protocol is FOUR stages, not three** (`train_ala.py`):
  1. χ VAMP-2 pretrain (`pre_train_epoch`, EarlyStopping patience=300)
  2. algebraic init `update_auxiliary_weights([probs, probs_tau], optimize_S=True)`
  3. **gradient U/S training `train_US()` with the lobe FROZEN** (`pre_train_epoch`
     epochs, patience=100) ← **WE ARE MISSING THIS STAGE**
  4. joint VAMPCE on all params (`epochs`, patience=200) after `set_optimizer_lr(0.2)`
  Step 5a wrongly concluded the algebraic init *replaced* the gradient-US phase and
  made it opt-in. Their code does **both**, in sequence.
- **`set_optimizer_lr` is defined NOWHERE** — not in their repo, not in deeptime
  (checked `DLEstimatorMixin`/`_vampnet.py` in a local deeptime clone). Their
  published `train_ala.py:264` / `train_ab.py:277` would raise AttributeError
  right before the joint phase ⇒ **the published code cannot run to completion
  as-is**, and the joint-phase lr is NOT recoverable from their source. Our
  `lr_all=1e-4` (reading 0.2 as a factor on 5e-4) is a reasonable GUESS, not fact.

### χ-plateau is probably NOT the blocker (revised)

χ VAMP-2 ~2.8 is robust across every variable tested — encoder variant v1 vs **v2**
(2.85 vs 2.83), n_neighbors 5/9, batch 200/1000, and seeds 1–4 (2.83–2.86). Since
their 4.41 is measured after the LONG joint VAMPCE phase (up to 1000 epochs,
patience 200) which also trains the lobe, a ~2.8 χ pretrain plateau is plausibly
normal. **Our joint phase only ever ran 30 epochs at lr 1e-4** → it never moved.
NEXT: implement missing stage 3 + run a long joint phase before concluding anything.

### Still true / next
- Aβ42 full run is long (~1.26M frames × 1300 epochs) — time one seed first,
  AFTER the alanine χ ceiling is understood (same χ machinery).
