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

| System | SchNet-χ / rev impl | smoke | 10-seed run | result |
|---|---|---|---|---|
| Alanine | ✓ implemented + CPU-smoke validated (chi→init→joint→PIPELINE COMPLETED) | ✓ (2–3 ep, strided) | ☐ not run | — |
| Aβ42 | ✓ script ready | ☐ | ☐ not run | — |

### Open before full runs (see REVGRAPHVAMP_TODO.md)
- GPU smoke on **real (unstrided)** alanine, few epochs, to confirm a
  non-degenerate VAMP-2 (the CPU smoke used heavy stride + 2–3 epochs → numbers
  meaningless, only the code path was validated).
- Confirm exact `pre_train_epoch`/`epochs` for alanine from their run command
  (Aβ42 has GitHub defaults 300/1000).
- Aβ42 full run is long (~1.26M frames × 1300 epochs) — time one seed first.
