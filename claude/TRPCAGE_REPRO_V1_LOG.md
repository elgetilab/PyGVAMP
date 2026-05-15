# TRPCAGE_REPRO_V1_LOG.md — Trp-cage reproduction v1 (dual-scoring baseline)

Companion to the villin VILLIN_REPRO_V*_LOG.md series. Target —
Ghorbani et al. 2022, Table S1:

> Trp-cage VAMP-2 = **4.79 ± 0.01** (10-seed average of per-batch
> averaged val VAMP-2, on DESHAW 2JOF).

## Why this run exists

The villin v11 array (`claude/VILLIN_REPRO_V11_LOG.md`) closed the
v1–v9 architecture-vs-attention story and demonstrated the
dual-scoring evaluator end-to-end. Result on villin:

| Metric | v11 (10 seeds) | Paper (10 seeds) |
|---|---|---|
| best concat | 3.7213 ± 0.0357 | — |
| perbatch_mean (paper methodology) | **3.6923 ± 0.0458** | **3.78 ± 0.02** |
| Δ | −0.0877 | — |

Within our cross-seed σ (1.9σ) though outside the paper's tighter
0.02σ (4.4σ). Reproduction is "close" by our criterion but not
identical. Moving to the next reproduction target — Trp-cage — to
see whether the same architecture + protocol lands within paper's
error bar on a different system.

## Architecture and protocol

Identical to villin v11. **No new flags, no new code.** Per
`feedback_per_experiment_presets`, the protein-specific values live
in this log + the SLURM script, not a new preset class.

| | Trp-cage v1 |
|---|---|
| Encoder | `schnet`, v11 corrected attention (`--use_attention`) |
| `--no_use_embedding` | yes (matches v11) |
| Classifier | `--clf_num_layers 1`, `--clf_dropout 0`, `--clf_norm none` |
| Init | `--init_method xavier_normal` |
| hidden_dim / output_dim | 16 / 16 |
| n_interactions | 4 |
| gaussian_expansion_dim | 16 |
| **n_neighbors** | **7** (vs villin 10 — per Ghorbani 2022 Table S1) |
| **n_states (k)** | **5** (vs villin 4) |
| **lag time** | **20 ns** (same as villin) |
| n_atoms (Cα) | 20 (vs villin 35) |
| **timestep** | **0.2 ps/frame** (DESHAW DCD metadata lies; actual = 200 ps/frame) |
| epochs | 100 |
| batch_size | 1000 |
| lr | 5e-4 |
| weight_decay | 1e-5 |
| val_split | 0.3 |
| `--no_discover_states` | yes (strict reproduction, no auto-k) |
| `--max_retrains 0` | yes |
| `--cache` | yes |

Dataset:
- **DESHAW trajectory:** 2JOF (Trp-cage, Lindorff-Larsen 208 µs)
- `/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/`
- 13 DCD files (per `tar -tJf` on the source archive)
- `topol.pdb` must be prepared separately (see "Data prep" below)

## Data prep (done 2026-05-12)

The 2JOF tar was extracted to `/mnt/hdd/data/trpcage/`:

```bash
mkdir -p /mnt/hdd/data/trpcage
tar -xJf /mnt/hdd/data/DESHAW/DESRES-Trajectory_2JOF-0-c-alpha.tar.xz \
    -C /mnt/hdd/data/trpcage
```

**mdtraj 1.11 cannot read `.mae` topology files** (supported topology
formats: `.pdb`, `.h5`, `.lh5`, `.prmtop`, `.parm7`, `.prm7`, `.psf`).
Both `md.load_topology(mae)` and `md.load(dcd, top=mae)` fail with
`OSError: detected ".mae" format is not supported`. So we parsed the
`.mae` atom table directly and emitted a PDB v3.3-conformant file:

```python
# /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb
# Built by regex-parsing 2JOF-0-c-alpha.mae's m_atom[20] section
# (atom name, residue name, residue number, x/y/z coords).
# 20 CA atoms, sequence DAYAQWLADGGPSSGRPPPS (TC10b variant).
```

Verified mdtraj round-trip:
- `md.load_topology(topol.pdb)` → 20 atoms, 20 residues, CA-only.
- `md.load(2JOF-0-c-alpha-000.dcd, top=topol.pdb)` → 100k frames × 20 atoms.

Note on the sequence: this is the TC10b "double mutant" Trp-cage variant
used in DESHAW 2JOF (Lindorff-Larsen 2011 / Ghorbani 2022), not the
canonical 1L2Y sequence `NLYIQWLKDGGPSSGRPPPS`.

## Probe (job 517, 2026-05-12 13:18 → 19:18 CEST, timeout)

`cluster_scripts/trpcage_repro_v1_probe.sh` — single seed-0 run on
`--gres=shard:1` + 8 CPUs + 64 GB + `--time=6:00:00` to confirm the
v11-size model fits in ~4 GB VRAM on the new sharded GPU layout.

**Result:**
- Training **completed all 100 epochs** in ~5h 58min on shard:1.
  Loaded best model at **concat = 4.6462** (epoch ~87, "no improvement
  for 13 epochs" reported at epoch 100). `best_model.pt` and
  `vamp_scores.png` saved.
- **Phase 3 (analysis) was killed** by SLURM at the 6h wall-time
  boundary, partway through dataset loading for inference. No CK test,
  no ITS, no attention maps, no interactive report.
- **No OOM:** GPU 0 stayed well under 4 GB for the single shard. The
  sharded GPU layout works.

**Implication for the array:** training fits comfortably in 6h, but
analysis needs another ~30–60 min. Wall time must be raised to give
analysis room (or `--time=INFINITE` since the partition allows it).

## Submission (job 522, 2026-05-12 23:34 onward, --array=0-9%8)

```bash
sbatch --array=0-9%8 cluster_scripts/trpcage_repro_v1_array.sh
```

Per-job SBATCH directives in `cluster_scripts/trpcage_repro_v1_array.sh`:
- `--gres=shard:1` (~4 GB VRAM per seed)
- `--cpus-per-task=2` (16-CPU partition cap / 8 concurrent)
- `--mem=16G` (128 GB partition cap / 8 concurrent)
- `--time=INFINITE` (training runs to completion, not time-gated)
- `--partition=gputraining`

**Prerequisite cluster change:** `MaxCPUsPerNode` raised from 8 to 16
in `HuginSLURM/config/slurm.conf` for the `gputraining` partition.
Reapplied via `sudo ./deploy.sh config` + `sudo systemctl restart
slurmctld slurmd`. Verified with `scontrol show partition gputraining`.

**Concurrency rationale:** GPU 0's 8 shards are always free; GPU 1's 8
shards are blocked 06:00–02:00 by vLLM. With ~7–10h training per seed,
GPU 1 shard jobs would overrun the vLLM restart window. So we stay on
GPU 0 → 8 concurrent maximum. Seeds 8–9 start automatically as wave-1
slots free up (`%8` throttle).

Outputs:
- Per-seed: `/mnt/hdd/experiments/trpcage_repro_v1/seed_NN/exp_trpcage_<TIMESTAMP>/`
- Logs: `/mnt/hdd/experiments/logs/trpcage_v1_<JOBID>_<SEEDID>.{out,err}`

## Throughput observation (mid-array, 2026-05-13 12:00)

At 12h 26min elapsed, all 8 wave-1 seeds were at epochs 47–50/100 →
**~15 min/epoch at 2 CPUs/job**. The probe ran at **3.5 min/epoch at
8 CPUs/job** — a ~4× per-seed slowdown that scales linearly with CPU
count.

GPU 0 utilization: **5% across all 8 concurrent jobs combined**, with
~12 GB / 32 GB VRAM in use. The reproduction protocol is severely
CPU-bound on data preparation, not GPU-bound on training compute.

This is structural to the v11 small-architecture reproduction
(hidden_dim=16, 20 atoms): GPU compute per batch is microseconds,
while CPU-side graph construction / batch assembly dominates. The
8-way concurrent shard:1 design is *the right call* for this
workload — we can't utilize more GPU per job, only more jobs.

Future systems with larger models (encoder sweeps, Aβ42 reversible)
will likely flip the bottleneck to GPU and want whole-GPU allocations
instead.

Investigation logged separately at
`claude/PYG_GRAPH_PREBUILD_INVESTIGATION.md` — pre-building PyG graph
objects into the dataset cache is one candidate to lift the ceiling
on small-model reproduction throughput. Not in scope for this array.

## Aggregation

After the array completes:

```bash
python cluster_scripts/aggregate_trpcage_v1_array.py \
    --csv /mnt/hdd/experiments/trpcage_repro_v1/summary.csv
```

The aggregator clones `aggregate_villin_v11_array.py` with three
changes:
- default `--root` → `/mnt/hdd/experiments/trpcage_repro_v1`
- default `--paper-mean` → 4.79
- default `--paper-sigma` → 0.01
- log-glob → `exp_trpcage_*/logs/log_*.txt`

It prints per-seed best_concat / perbatch_mean@best_epoch and
cross-seed mean ± stdev for both, with Δ vs paper in both σ
conventions.

## Comparison to villin v11

| | Villin v11 (10 seeds) | Trp-cage v1 (10 seeds, expected) |
|---|---|---|
| n_atoms (Cα) | 35 | 20 |
| n_neighbors | 10 | 7 |
| n_states | 4 | 5 |
| Paper VAMP-2 | 3.78 ± 0.02 | 4.79 ± 0.01 |
| Best concat (ours) | 3.7213 ± 0.0357 | TBD |
| perbatch_mean@best (ours) | 3.6923 ± 0.0458 | TBD |
| Δ vs paper | −0.0877 | TBD |

If the Trp-cage Δ is comparable in magnitude and sign to villin's,
the gap is a system-independent property of our architecture +
protocol vs the paper's (the prime suspect would be an architectural
or hyperparameter detail we're still missing across systems, not a
villin-specific issue). If the Trp-cage Δ is meaningfully smaller or
flips sign, villin has system-specific issues to revisit.

## Result (job 522, 2026-05-12 23:34 → 2026-05-15 ~02:00 CEST, all 10 seeds ok)

Wave 1 (seeds 0–7, %8 concurrent): ~25h elapsed each, finished
~00:30 May 14. Wave 2 (seeds 8–9, 2 concurrent): ~26h each, finished
~02:00 May 15. Both waves ran training + analysis end-to-end. No
errors anywhere; all 10 seeds reported `ok` by the aggregator.

### Headline numbers (cross-seed, n=10)

| Metric | Trp-cage v1 | Paper |
|---|---|---|
| best concat | **4.6564 ± 0.0239** | — |
| perbatch_mean @ best-concat epoch | **4.6516 ± 0.0175** | **4.79 ± 0.01** |
| Δ vs paper (perbatch) | **−0.1384** | — |
| ↳ in paper's σ | 13.8σ | — |
| ↳ in **our** σ | **7.9σ** | — |

### Per-seed table

| seed | best epoch | best concat | perbatch_mean | perbatch_std |
|---:|---:|---:|---:|---:|
| 0  | 86 | 4.6446 | 4.6393 | 0.1431 |
| 1  | 95 | 4.6501 | 4.6450 | 0.1303 |
| 2  | 96 | 4.6638 | 4.6545 | 0.1886 |
| 3  | 75 | 4.6366 | 4.6317 | 0.1326 |
| 4  | 83 | 4.6520 | 4.6509 | 0.1247 |
| 5  | 91 | 4.6366 | 4.6399 | 0.1239 |
| 6  | 98 | 4.6467 | 4.6481 | 0.1316 |
| **7** | **87** | **4.7175** | **4.6920** | 0.1921 |
| 8  | 91 | 4.6466 | 4.6444 | 0.1291 |
| 9  | 95 | 4.6692 | 4.6699 | 0.1626 |

Best seed (concat & perbatch): seed_7 = 4.7175 / 4.6920. Closest any
seed got to paper's 4.79: still −0.10 below in perbatch.

CSV at `/mnt/hdd/experiments/trpcage_repro_v1/summary.csv`.

### Verdict — **not parity**

- Δ = −0.1384 is **outside paper's σ by 13.8×** (their ± 0.01 is very
  tight) AND **outside our own cross-seed σ by 7.9×** (ours is
  ± 0.0175 — equally tight). The gap is therefore not seed noise on
  our side.
- For comparison, villin v11's Δ = −0.0877 sat inside our cross-seed
  σ (1.9σ) — "close but not identical". Trp-cage's gap is
  qualitatively worse: bigger absolute, and clearly outside scatter.

### Cross-system pattern (villin + Trp-cage)

Both systems undershoot the paper in the same direction. With two
data points the natural reading is **systematic protocol/architecture
detail still missing from our reproduction**, not a per-system issue.

| | Villin v11 (10 seeds) | **Trp-cage v1 (10 seeds)** |
|---|---|---|
| n_atoms / n_neighbors / n_states | 35 / 10 / 4 | 20 / 7 / 5 |
| Paper VAMP-2 | 3.78 ± 0.02 | 4.79 ± 0.01 |
| Best concat (ours) | 3.7213 ± 0.0357 | 4.6564 ± 0.0239 |
| perbatch_mean (ours) | 3.6923 ± 0.0458 | 4.6516 ± 0.0175 |
| Δ vs paper | −0.0877 | **−0.1384** |
| Δ in our σ | 1.9σ (within scatter) | **7.9σ (outside scatter)** |

The `project_villin_gap` memory has the τ-normalization hypothesis as
the leading candidate for villin's residual gap. If the same
mechanism is at play on Trp-cage, the larger absolute Δ would need
to be explained by Trp-cage's different timescale structure — worth
testing once a gap-investigation lane is opened.

## Followups

- **Hold on NTL9 / Aβ42** until the gap pattern is understood, per
  user 2026-05-15 decision (rigor over checklist progress). The two
  remaining reproduction targets will land more interpretably once we
  know whether the v11 architecture has a systematic offset.
- Gap-investigation candidates (when prioritized):
  - τ-normalization (per `project_villin_gap`)
  - Train/val split convention (random_split vs perm-based)
  - Batch-size or LR-schedule details vs Ghorbani 2022's actual code
  - Architecture details: e.g. activation choices in the InteractionBlock,
    BatchNorm placement, edge-attribute scaling
- `claude/PYG_GRAPH_PREBUILD_INVESTIGATION.md` — open, orthogonal to
  the gap question; would meaningfully cut wall time on any
  small-architecture reproduction.
