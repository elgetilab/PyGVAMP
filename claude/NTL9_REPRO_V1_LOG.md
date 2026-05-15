# NTL9_REPRO_V1_LOG.md — NTL9 reproduction v1 (vectorized + dual-scoring)

Third reproduction system after villin v11 and Trp-cage v1. Target —
Ghorbani et al. 2022, Table S1:

> NTL9 VAMP-2 = **4.59 ± 0.09** (10-seed average of per-batch averaged
> val VAMP-2, on DESHAW NTL9 — 1.11 ms total).

## Why this run exists

Two reproduction results so far:

| | Villin v11 (10 seeds) | Trp-cage v1 (10 seeds) | Paper σ |
|---|---|---|---|
| Δ vs paper | −0.0877 (1.9σ ours) | **−0.1384 (7.9σ ours)** | tight (± 0.02, ± 0.01) |

Both undershoot the paper in the same direction. With NTL9's wider
paper σ (± 0.09 — 9× looser than Trp-cage's ± 0.01), a similar Δ
could still land inside the paper's bar. NTL9 is therefore the most
informative reproduction next:

- **If NTL9 lands within paper σ** (e.g. Δ ≈ −0.10 inside ± 0.09):
  systematic gap is real but small; we likely call villin + Trp-cage
  "close" and ship the reproduction grid with that caveat.
- **If NTL9 also undershoots well outside paper σ**: cross-system
  systematic offset is robust; warrants a gap-investigation lane
  before publishing reproduction claims.

## What's different from Trp-cage v1

**Architecture: unchanged.** Same v11 corrected-attention small-SchNet
+ dual-scoring eval. Per `feedback_per_experiment_presets`, the
protein-specific values live in the SLURM script + this log, not a
new preset class.

**Code: PR #10 graph-build vectorization merged.** This is the first
production array using the new pipeline. Speedtest job 533 confirmed
~5.5× per-epoch speedup vs the Trp-cage baseline (15 min/epoch → ~2.7
min/epoch at 2 CPUs/job).

## Architecture and protocol

| | NTL9 v1 |
|---|---|
| Encoder | `schnet`, v11 corrected attention (`--use_attention`) |
| `--no_use_embedding` | yes |
| Classifier | `--clf_num_layers 1`, `--clf_dropout 0`, `--clf_norm none` |
| Init | `--init_method xavier_normal` |
| hidden_dim / output_dim | 16 / 16 |
| n_interactions | 4 |
| gaussian_expansion_dim | 16 |
| **n_neighbors** | **10** (per Ghorbani 2022 Table S1) |
| **n_states (k)** | **5** |
| **lag time** | **200 ns** (10× larger than villin/Trp-cage's 20 ns) |
| n_atoms (Cα) | 39 |
| **timestep** | **0.2 ps/frame** (DESHAW DCD metadata lies; actual = 200 ps/frame) |
| epochs | 100 |
| batch_size | 1000 |
| lr | 5e-4 |
| weight_decay | 1e-5 |
| val_split | 0.3 |
| `--no_discover_states` | yes |
| `--max_retrains 0` | yes |
| `--cache` | yes |

## Data prep (done 2026-05-15)

Four DESHAW NTL9 archives extracted to `/mnt/hdd/data/ntl9/`:

```bash
mkdir -p /mnt/hdd/data/ntl9
for i in 0 1 2 3; do
  tar -xJf /mnt/hdd/data/DESHAW/DESRES-Trajectory_NTL9-${i}-c-alpha.tar.xz \
      -C /mnt/hdd/data/ntl9 &
done
wait
```

**149 DCD files total** (NTL9-0: 56, NTL9-1: 54, NTL9-2: 20, NTL9-3: 19),
all 39 atoms each, 200 ps/frame. The pipeline's recursive glob with
`--file_pattern 'NTL9-*-c-alpha-*.dcd'` matches all of them in one
sweep.

**`topol.pdb` built** the same way as trpcage — regex-parse the .mae
`m_atom[39]` section (since mdtraj 1.11 can't read `.mae`), emit
PDB v3.3-conformant CA-only file at `/mnt/hdd/data/ntl9/topol.pdb`.
Sequence: `MKVIFLKDVKGMGKKGEIKNVADGYANNFLFKQGLAIEA` (39 residues,
NTL9 canonical). Topology is shared across all 4 trajectories (same
protein).

Verified mdtraj round-trip + DCD load against all 4 trajectories:
each loads 100k frames per DCD × 39 atoms cleanly.

## Module rebuild required before submission

The installed `/opt/software/pygvamp/1.0.0/source/pygv/dataset/vampnet_dataset.py`
has mtime **May 7**, predating the PR #10 vectorization (May 14). The
speedtest worked around this with a `PYTHONPATH=$LOCAL_PYGV:$PYTHONPATH`
override, but for a 10-seed production array a clean rebuild is
safer.

```bash
sudo bash /home/vi/PycharmProjects/PyGVAMP/module/install_module.sh \
    --prefix /opt/software/pygvamp/1.0.0 \
    --moduledir /opt/modulefiles \
    --skip-env
```

Verify after rebuild:

```bash
module purge && module load pygvamp/1.0.0
python -c "
import inspect
from pygv.dataset.vampnet_dataset import VAMPNetDataset
src = inspect.getsource(VAMPNetDataset._create_graph_from_frame)
print('Vectorized' if 'for i in range(self.n_atoms)' not in src else 'OLD CODE')
"
```

Should print `Vectorized`.

## Submission

```bash
sbatch --array=0-9%8 cluster_scripts/ntl9_repro_v1_array.sh
```

Per-job SBATCH directives in `cluster_scripts/ntl9_repro_v1_array.sh`:
- `--gres=shard:1` (~4 GB VRAM per seed)
- `--cpus-per-task=2`, `--mem=16G`
- `--time=INFINITE` (training runs to completion)
- `--partition=gputraining`

**Concurrency:** 8 concurrent on GPU 0 shards. Seeds 8–9 start
automatically as wave-1 slots free (`%8` throttle). Stays clear of
GPU 1 / vLLM, same as Trp-cage array.

## Wall-time estimate

- Trp-cage v1 (vectorized speedtest): ~2.7 min/epoch at 2 CPUs, 970
  train batches/epoch.
- NTL9 data scale: ~5.5M frames × val_split=0.3 → ~3850 train
  batches/epoch (~4× Trp-cage).
- Linear projection: ~10–12 min/epoch → ~17–20h per seed at 100 epochs.
- 8 concurrent first wave: ~17–20h. Seeds 8–9 then run with 2
  concurrent → ~17–20h more. **Total: ~35–40h** end-to-end.

(For comparison: Trp-cage v1 at the old code rate took ~50h. Even
though NTL9 has 4× more batches per epoch, the vectorization
recoups enough to keep total wall time in a similar ballpark.)

## Aggregation

```bash
python cluster_scripts/aggregate_ntl9_v1_array.py \
    --csv /mnt/hdd/experiments/ntl9_repro_v1/summary.csv
```

Clone of `aggregate_trpcage_v1_array.py` with: default `--root` →
`/mnt/hdd/experiments/ntl9_repro_v1`, `--paper-mean 4.59`,
`--paper-sigma 0.09`, log-glob `exp_ntl9_*/logs/log_*.txt`.

## Result

_To be filled in after the array completes._

### Headline numbers (cross-seed, n=10)

- Best concat: _TBD_
- perbatch_mean @ best-concat epoch: _TBD_
- Δ vs paper (4.59 ± 0.09): _TBD_

### Per-seed table

_(filled by aggregator output)_

### Verdict

_TBD — three branches:_
- _**Inside paper σ** (Δ within ± 0.09): close enough, mark
  reproduction "achieved" with the noted cross-system systematic
  offset._
- _**Outside paper σ, inside our σ**: like villin — borderline,
  document as close-but-not-identical._
- _**Outside both σ**: like Trp-cage — clear miss, lock the
  gap-investigation lane before further reproductions._

## Followups

- If NTL9 lands inside paper σ:
  - Aβ42 reversible reproduction (RevVAMPNet path, different model
    class).
  - Then encoder improvement sweeps (GIN / ML3).
- If NTL9 misses outside both σ:
  - Open the cross-system gap investigation. Candidates per
    `project_villin_gap` and the Trp-cage log:
    - τ-normalization
    - train/val split convention (random_split vs perm-based)
    - LR-schedule / batch-size details vs Ghorbani 2022's code
    - InteractionBlock activation / BN placement
- Either way: `claude/PYG_GRAPH_PREBUILD_INVESTIGATION.md` remains
  open and orthogonal — Option C (vectorize kNN loop) was already
  taken in PR #10; pre-building edges into the cache is the next
  speedup if needed.
