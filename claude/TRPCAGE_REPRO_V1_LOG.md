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

## Result

_To be filled in after the array completes._

### Headline numbers

- Best concat (cross-seed): _TBD_
- perbatch_mean @ best concat epoch (cross-seed): _TBD_
- Δ vs paper (4.79): _TBD_

### Per-seed table

_(filled by aggregator output)_

### Verdict

_TBD — `parity-claimed` if cross-seed perbatch overlaps 4.79 ± 0.01
in either σ convention. `not-yet` if Δ > 0.1 with no overlap._

## Followups

- If parity: move to NTL9 (k=5, τ=200 ns, paper 4.59 ± 0.09).
- If not parity: cross-system gap study (villin Δ + Trp-cage Δ
  together inform whether to investigate architecture or call it a
  "close-but-not-identical" reproduction).
- Either way: aBeta42 reversible reproduction next (different model
  class — RevVAMPNet — first time exercised end-to-end at scale).
