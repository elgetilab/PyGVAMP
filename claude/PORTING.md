# Porting PyGVAMP experiments to another workstation

Everything in `cluster_scripts/` was written against one machine (`hugin`). None of
it is portable as-is. This lists every assumption that will break, why, and what to
check — ordered by how badly it fails.

## The reference machine (hugin), for comparison

| | |
|---|---|
| GPUs | 2 × RTX 5090, 32,607 MiB each |
| GPU sharding | SLURM `shard` GRES, **8 shards/GPU = 4,076 MiB per shard** |
| GPU 1 | **blocked 06:00–02:00 by a vLLM service** — effectively 1 usable GPU |
| CPU / RAM | 24 cores / 249 GB |
| Partition | `gputraining`, `MaxTime=UNLIMITED`, `PreemptMode=OFF` |
| Partition caps | mem 128000M, 16 CPU |
| `sacct` | **disabled** — job history only via `scontrol show job` and logs |
| Data | `/mnt/hdd/data/` (22 TB) |
| Experiments | `/mnt/hdd/experiments/`, logs in `/mnt/hdd/experiments/logs/` |
| Module | `pygvamp/1.0.0` at `/opt/software/pygvamp/1.0.0`, modulefile `/opt/modulefiles` |
| Stability | hard-crashes with no shutdown record; was off Fri→Mon at least once |

## 1. `--gres=shard:N` — silently wrong elsewhere (WORST)

`shard` is **not** a standard SLURM GRES; it comes from a local sharding config, and
**a shard is a fraction of a specific GPU**. `shard:3` means 12.2 GB *on hugin* and
means nothing (or something different) anywhere else. On a machine without shards
the jobs will not even be schedulable.

Replace with whatever the new machine uses — typically `--gres=gpu:1`, or MPS, or
MIG slices.

**Then re-derive concurrency from measured memory, not from the old numbers.** The
α3D arm failed exactly here: `shard:2` (8.15 GB) was requested for a job whose
measured peak was 7.7 GB, leaving no room for the CUDA context, and **two arrays
each submitted at `%2` do not compose to 2 concurrent** — they gave 4, ~31 GB on a
32 GB GPU, and all 20 tasks died with `CUDA out of memory in encoder` **while
exiting 0**. Express the limit in the *resource request* so the scheduler enforces
it GPU-wide; array `%throttle` does not compose across submissions.

Measured peaks for sizing: Trp-cage (20 CA, 1.04M frames) fits a single shard;
α3D (73 CA, 176k samples after stride) peaks at **7.7 GB**.

## 2. Hardcoded absolute paths

`/mnt/hdd/data/...`, `/mnt/hdd/experiments/...` appear in every script (40+ log-path
references alone). There is no path variable to override. Either recreate the layout
or sed the scripts. Data locations currently assumed:

```
/mnt/hdd/data/{trpcage,villin,ntl9,gtt,ab42,a3d}/
/mnt/hdd/data/DESHAW/            # 49 DESRES tarballs, 12 fast-folder systems
/mnt/hdd/data/painn_candidates/  # extracted A3D-0/1, NuG2-0, lambda-0
```

## 3. The module vs the repo — a recurring foot-gun

Jobs run the **installed module**, not the working tree:
`module load pygvamp/1.0.0` → `/opt/software/pygvamp/1.0.0/source/pygv` (an editable
install). Consequences:

- New CLI flags do not exist until the module is redeployed:
  `sudo module/install_module.sh --prefix <prefix> --moduledir <moduledir> --skip-env`
  (`--skip-env` reuses the conda env; without it PyTorch/PyG rebuild from scratch).
- It deploys **committed `HEAD`** (`git archive HEAD`), not the working tree.
- `PYGVAMP_SRC_OVERRIDE=<repo>` sets `PYTHONPATH` and wins over the module — the
  escape hatch when the module lags. Verified to take precedence because
  setuptools' editable finder appends to `sys.meta_path`.
- **Verify from OUTSIDE the repo directory.** `python -c "import pygv"` run inside
  the repo puts CWD first on `sys.path` and reports the repo, not the module. This
  produced a false "module is fine" reading once.

Scripts written after 2026-08-03 carry a **preflight** that greps `pygvamp --help`
for a required flag and exits with a readable message rather than an argparse dump
across every array task. Keep that pattern.

## 4. Requeue survival — needed because hugin crashes

`--exp_name`, `--resume_training`, `--save_every`, plus `#SBATCH --requeue` and
`--open-mode=append`. On a stable machine these are optional, but two are worth
keeping regardless:

- **`--open-mode=append`** — without it every requeue *truncates* the `.out`,
  destroying the evidence from earlier attempts.
- **`--exp_name`** — a deterministic experiment directory. Without it each attempt
  mints `exp_<protein>_<timestamp>` and redoes prep + training from epoch 0. Job 877
  lost 11 attempts across 4 lags to exactly this.

Completion is decided by **marker files** (`training_complete.json`,
`analysis_complete.json`), never by the presence of `best_model.pt` — that file is
written on every validation improvement from epoch 1, so an interrupted run leaves
one behind. Re-submitting a whole array is the correct way to resume: finished
seeds are skipped by their markers.

## 5. Exit 0 does not mean results exist — audit every campaign

Four distinct failures in this project exited 0 with complete-looking output. Run
this before computing any statistic:

```bash
# per run
python -c "import json;d=json.load(open('<exp>/pipeline_summary.json'));print(d['analysis_completed'])"
grep -c 'NaN values detected'  <log>   # must be 0 — NOT caught by grep Error|Traceback
grep -c 'out of memory'        <log>   # must be 0
grep    'retrain loop exhausted' <log> # when k* is being measured
# and: no seed at the degenerate score (VAMP-2 == 1.0 exactly == rank-1)
```

The four: swallowed Phase-3 exception (τ=50 stride); no-attention encoder killing
analysis; `init_for_vamp` driving PaiNN to NaN, masked to VAMP-2 = 1.0000 with
**37,762 NaN warnings**; and the OOM above.

## 6. Data acquisition on the new machine

DESRES trajectories are licensed and not redistributable — re-download rather than
copy if the licence requires it. `/mnt/hdd/data/DESHAW/` holds 49 tarballs covering
12 Lindorff-Larsen fast folders (1FME, 2F4K, 2JOF, 2WAV, A3D, CLN025, GTT, lambda,
NTL9, NuG2, PRB, UVF), each in `-c-alpha` and `-protein` variants.

**Topology trap:** `system.mae` in the distribution root is the **full solvated
system** (25,921 atoms for A3D) and `mae_to_topol_pdb.py` will happily produce
garbage from it. Use the **reduced structure file inside the DCD directory**
(e.g. `A3D-0-c-alpha.mae`). Always check the converted atom count against the DCD:

```python
import mdtraj as md
with md.formats.DCDTrajectoryFile(f) as fh: print(fh.read(n_frames=1)[0].shape[1])
```

**`--timestep 0.2` is mandatory for every DESRES set here** — the DCD metadata
reports 1 ps but the physical step is 200 ps. Omitting it puts every lag off by 200×.

## 7. Cost model — re-measure, do not port

Wall time scales with **frame count**, not protein size (Trp-cage: 20 CA but 1.04M
frames at stride 1 → ~30 h/seed; α3D: 73 CA but 176k samples → ~20–35 min/seed).
**Analysis dominates training** and scales with k.

Standing rule from three wrong GTT estimates: **run one full seed and read the wall
time before sizing a campaign.** Also note per-epoch rates degrade with concurrency
(α3D PaiNN: 20 s/epoch alone, ~50 s at 2-way).

## 8. Test baseline — the suite is fully green

`pytest tests/ -q` → **744 passed, 0 failed, 9 skipped, 2 xfailed**.
**Any failure on the new machine is a porting problem, not a pre-existing one.**

Wall time is dominated by two disk-bound integration files that read real MD data
(~58 min total here, and it varies a lot with I/O). The other ~700 tests run in
~13 s:

```bash
pytest tests/ -q -m "not integration" \
    --ignore=tests/test_phase5_integration.py \
    --ignore=tests/test_pipeline_integration.py     # ~13 s smoke check
```

Those two integration files hardcode data paths — they are the first thing to
repoint on a new machine.

**Meta / MetaAtt** were broken until 2026-08-25 (the fallback for `torch_scatter`
exported only a subset of what they call, and `scatter_min/max/mul` inside the
"fallback" called `torch.ops.torch_scatter.*` — requiring the very extension they
substitute for). Now fixed and pinned by `tests/test_scatter_fallback.py`, which
checks the contract structurally so it cannot drift. **The crash is fixed; the
encoder is not validated** — Meta still warns that it is experimental, has never
appeared in any comparison, and would need its own baseline before use as an arm.

## Where the results live

| lane | document |
|---|---|
| Category 1 reproductions | `experiments/{trpcage,villin,ntl9}_*.md`, `experiments/revgraphvamp_repro.md`, `experiments/ab42_{red,ox}.md` |
| Category 2 encoders (CLOSED) | `experiments/{trpcage,villin,ntl9,a3d}_encoders.md` |
| Category 3 lag exploration | `experiments/gtt_lagsweep.md` |
| PaiNN design + measurements | `claude/PAINN_SCOPE.md` |
| Requeue/resume design | `claude/RESUME_PLAN_2026-08-03.md` |
| **All VAMP-2 numbers, one page** | **`RESULTS.md`** (start here) |
| Master status | `claude/EXPERIMENT_CHECKLIST.md` |
