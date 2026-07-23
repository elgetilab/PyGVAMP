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

Remaining hypotheses (NOT yet tested — need a decision on direction):
- (A) **Table-2 4.41 is their graph-ATTENTION model, not SchNet.** "RevGraphVAMP"
  = NeighborMultiHeadAttention encoder; SchNet is likely a baseline that scores
  lower. Matching 4.41 may require porting their attention encoder (we have
  schnet/gin/ml3, not NeighborMultiHeadAttention). ← strongest hypothesis.
- (B) **Encoder-detail mismatch within SchNet path**: their GraphVampNet has
  residual conns, attention/AA pooling, atom-embedding init, dropout=0.4, h_g=8
  projection — options our SchNet-χ + classifier may not replicate.
- (C) **State collapse mechanism** (3/6 states at 0%): could be seed-dependent;
  a diagnostic multi-seed χ sweep would show if any seed escapes ~2.9 (frame as
  diagnosis, NOT seed-cherry-picking — see user rigor pref).
- Their exact alanine command line is NOT published (README shows only Aβ42,
  data file `red_5nbrs_...` ⇒ Aβ42 uses 5 neighbors); can't just copy the recipe.

### Still true / next
- Aβ42 full run is long (~1.26M frames × 1300 epochs) — time one seed first,
  AFTER the alanine χ ceiling is understood (same χ machinery).
