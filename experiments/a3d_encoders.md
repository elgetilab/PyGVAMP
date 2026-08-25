# alpha3D (A3D) — SchNet vs PaiNN — Tracker

Equivariant-encoder test on the campaign's only non-two-state system. Answers the
one axis the earlier encoder work never touched: not whether the Cα distance map
*contains* the information (the angular pre-test and the k-NN truncation
measurement both bounded that), but whether an explicit **equivariant inductive
bias** makes the same information easier to *learn*.

## System Info

| Property | Value |
|----------|-------|
| Protein | alpha3D, three-helix bundle (DESRES A3D, Lindorff-Larsen 2011) |
| Data | `/mnt/hdd/data/a3d/` → A3D-0 (18 dcd) + A3D-1 (19 dcd) = 37 chunks |
| Selection | `name CA` — **73 atoms** |
| Frames | ~3.47M total (~694 µs); 353,520 lagged pairs → 176,760 after auto-stride |
| Timestep | **`--timestep 0.2` MANDATORY** — times.csv shows 200 ps/frame; DESRES DCD metadata says 1 ps (200× error) |
| Lag / states | 50 ns / **k=3** |
| Split | 123,732 train / 53,028 val (val_split 0.3) |

**Topology gotcha.** `system.mae` in the distribution root is the **full solvated
system** (25,921 atoms) — the mae→pdb converter produces garbage from it. The
correct source is the *reduced* structure file inside the DCD directory
(`A3D-0-c-alpha.mae`), which gives 73 CA matching the DCD exactly. A3D-0 and A3D-1
topologies differ only in REMARK lines and reference coordinates; the residue
sequence is identical, so one topol.pdb serves both. They are **separate runs** and
are never concatenated.

## Why k=3 (probe, job 921)

Unlike every earlier system, alpha3D is **not two-state**. A discovery + JSD
retrain probe (SchNet, 1 seed, `--max_retrains 10`) descended
**10→9→8→7→6→5→4→3** and converged on a real verdict — `recommendation=keep`,
`confidence=high`, `Convergence: recommended k=3 matches trained n_states` — using
7 of 10 allowed rounds, so k* was measured, not truncated by the cap.

Populations at k*=3: **0.469 / 0.362 / 0.169** — three genuinely occupied states,
unlike GTT's 0.77/0.23 or the 0.4% vestigial sliver its capped τ=1 run produced.

| k | VAMP-2 | ceiling |
|---|---|---|
| 10 | 9.0016 | 10 |
| 5 | 4.9290 | 5 |
| 4 | 3.9365 | 4 |
| **3** | **2.9810** | 3 |

k=3 is then **pinned** for the comparison (discovery off, `--max_retrains 0`), because
k must be identical across arms or the comparison is confounded. This is the
opposite of the GTT setup, where the retrain loop was itself the quantity measured.

## Held fixed across BOTH arms (single-variable swap)

data, `name CA`, timestep 0.2, stride 10 + auto_stride, lag 50 ns, k=3,
`--no_discover_states --max_retrains 0 --no_warm_start_retrains`,
output_dim 16, n_interactions 4, n_neighbors 10, gaussian 16, no embedding,
lr 5e-4, epochs 100, batch 1000, val_split 0.3, seeds 0–9.
**The arms differ only in the encoder and its width.** This fixes the known
weakness of the GIN/ML3 table, which compared whole recipes rather than encoders.

### Capacity is parameter-matched, not width-matched

Encoder-only parameter counts (edge_dim 16, output_dim 16, n_interactions 4):

| | SchNet | PaiNN @ width 16 | ratio |
|---|---|---|---|
| node_dim=20 (Trp-cage) | 7,600 | 15,952 | **2.10×** |
| node_dim=73 (alpha3D) | 38,976 | 16,800 | **0.43×** |

SchNet's count scales with `node_dim` (one-hot over atoms); PaiNN projects to
`hidden_dim` immediately. So "equal width" **inflates** PaiNN 2× on Trp-cage and
**handicaps** it 2.3× here — a 5× swing. Matched setting used:
`--painn_hidden_dim 26` → **38,990 vs 38,976 (1.0004×)**.
**Recompute per system. Do not reuse 26.**

## Results (best-concat VAMP-2, ceiling 3.0)

| seed | SchNet | PaiNN | Δ |
|---|---|---|---|
| 0 | 2.9627 | 2.9649 | +0.0022 |
| 1 | 2.9721 | 2.9700 | −0.0021 |
| 2 | 2.9589 | 2.9581 | −0.0008 |
| 3 | 2.9611 | 2.9592 | −0.0019 |
| 4 | 2.9520 | 2.9673 | +0.0153 |
| 5 | 2.9639 | 2.9591 | −0.0048 |
| 6 | 2.9638 | 2.9635 | −0.0003 |
| 7 | 2.9724 | 2.9751 | +0.0027 |
| 8 | 2.9584 | 2.9554 | −0.0030 |
| 9 | 2.9675 | 2.9641 | −0.0034 |

| arm | n | mean ± sd | 95% CI | % of ceiling |
|---|---|---|---|---|
| **SchNet** | 10 | **2.9633 ± 0.0063** | [2.9594, 2.9672] | 98.78% |
| **PaiNN** | 10 | **2.9637 ± 0.0060** | [2.9599, 2.9674] | 98.79% |

- **Δ = +0.0004**, CIs essentially superimposed.
- **Paired t = 0.203** (|t| ≈ 2.26 needed at p=0.05, df 9). Paired is the correct
  test — both arms share seeds and therefore train/val splits.
- Paired 95% CI on the difference: **[−0.0032, +0.0039]**.
- Minimum detectable effect (paired, 80% power, α=0.05): **~0.0053**.
- Unpaired Welch t = 0.134 — same conclusion.
- **0 collapsed seeds** in either arm, 0 OOM, 0 NaN across all 20 runs.

**Conclusion: PaiNN ≡ SchNet on alpha3D at k=3. No effect larger than ~0.005.**
That is an order of magnitude tighter than the angular pre-test's ~0.02 bound.

### ⚠️ The ceiling caveat — weight this heavily

Both arms sit at **98.8% of the theoretical maximum (3.0)**. There is almost no
headroom in which any encoder *could* distinguish itself, so this null is partly a
statement about the **task**, not only about PaiNN. It was flagged before the
numbers came in, not after.

Supporting evidence that the encoder is not what drives variation: the **seed
effect dwarfs the encoder effect**. Seed 7 is highest in both arms (2.9724 /
2.9751), seed 8 near-lowest in both (2.9584 / 2.9554). Seed 4 is the only seed with
a Δ above noise (+0.0153) and it is a SchNet outlier (2.9520, its worst seed), not
a PaiNN gain.

## Two defects found and fixed by this experiment

Both produced **exit 0 with complete-looking output** — the recurring trap in this
project.

1. **PaiNN collapse via `init_for_vamp` (job 922).** `create_model` applied
   `init_for_vamp(model, 'kaiming_normal')`, which picks its scheme by sniffing
   module type names for `GCN|GAT|GraphConv|GIN|EdgeConv`. SchNet's
   `GCNInteraction` matches; PaiNN matches nothing and fell through to a path that
   blows up its residual scalar/vector accumulation. The forward produced NaN,
   VAMPNet's guard silently rewrote `NaN → 1e-6`, and the model trained 100 epochs
   at the degenerate **VAMP-2 = 1.0000**, exiting 0 with a complete analysis and
   **37,762 NaN warnings** that a `grep Error|Traceback` does not catch.
   Fixed: encoders declare `self_initialized`; `apply_vamp_init` preserves them.
   Had this not been caught, all 10 PaiNN seeds would have read 1.0000 and the
   natural conclusion would have been "PaiNN catastrophically fails on alpha3D" —
   a wrong scientific result from a one-line init bug.

2. **CUDA OOM from an under-requested GRES (jobs 925/926).** `shard:2` grants
   8.15 GB; measured peak is 7.7 GB, so the request left no room for the CUDA
   context. Worse, two arrays each submitted at `%2` do **not** compose to 2
   concurrent — they gave 4, ~31 GB on a 32 GB GPU. All 20 tasks died with
   `CUDA out of memory in encoder` and exited 0. Fixed by requesting `shard:3`,
   which is honest about usage and lets SLURM cap concurrency GPU-wide regardless
   of how many arrays are submitted.

## Costs (measured)

| | value |
|---|---|
| probe (8 models + 8 analyses, discovery + retrain) | 11h33m |
| prep with discovery | ~1h20m (Graph2Vec + clustering sweep dominates) |
| prep without discovery | ~1 min |
| SchNet training | ~12 s/epoch uncontended |
| PaiNN training | ~20 s/epoch uncontended (**1.67×**), ~50 s/epoch at 2-way |
| per seed at k=3 | SchNet ~20–35 min, PaiNN ~1h50m |
| GPU | 7.7 GB peak → **request shard:3** |

## Reproduce

```bash
# probe (k* determination)
sbatch cluster_scripts/a3d_probe.sh
# arms — no %throttle needed, shard:3 caps concurrency GPU-wide
A3D_ENCODER=painn  sbatch --array=0-9 cluster_scripts/a3d_encoder_array.sh
A3D_ENCODER=schnet sbatch --array=0-9 cluster_scripts/a3d_encoder_array.sh
```

Finished seeds are skipped by their completion markers, so re-submitting a whole
array is the correct way to resume after a partial failure.

## Open threads

- **The ceiling.** A discriminating operating point (larger k, or a shorter lag
  where states are less separable) would give an encoder room to differ. Not run —
  the prior after five negative results is that it would also come back flat.
- **k=3 rests on one seed.** The probe converged cleanly and well inside the cap,
  but k* seed-stability was not tested (~11.5h/seed to check).
- `COLLAPSED_preinitfix_exp_a3d_painn_s00/` is retained as the on-disk record of
  defect 1. Do not delete without also removing its completion markers — the
  resume logic would otherwise skip a rerun and silently reuse the degenerate model.
