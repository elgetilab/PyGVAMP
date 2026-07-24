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

## ⛔ BLOCKER (found 2026-07-23) — χ-stage VAMP-2 ceilings at ~2.9 on alanine

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
