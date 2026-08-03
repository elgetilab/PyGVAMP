# Requeue-resumable runs — investigation + plan (2026-08-03)

Context: GTT lag sweep (job 877) burned 11 attempts across 4 lags and produced
0 usable results. Every restart redoes prep + cache + training from epoch 0.

---

## Part A — why the jobs restart

### It is NOT cluster time, and NOT preemption

| check | value |
|---|---|
| `scontrol show partition gputraining` | `MaxTime=UNLIMITED`, `PreemptMode=OFF`, `GraceTime=0` |
| script request | `#SBATCH --time=INFINITE` |
| reservations | none |
| `scontrol show config` | `PreemptType=(null)`, `OverTimeLimit=0` |

No job has ever hit a wall clock limit. `sacct` is disabled on this cluster, so
job history has to be read from `scontrol show job` + the logs.

### The node is hard-crashing

`journalctl --list-boots` — boot boundaries since the sweep started:

```
-4  Mon 2026-07-27 10:55:00  ->  Thu 2026-07-30 15:30:18   crash 1
-3  Thu 2026-07-30 15:31:29  ->  Thu 2026-07-30 20:01:45   crash 2
-2  Thu 2026-07-30 20:02:31  ->  Fri 2026-07-31 04:39:24   crash 3
-1  Fri 2026-07-31 04:40:12  ->  Fri 2026-07-31 07:25:53   crash 4
 0  Mon 2026-08-03 07:46:39  ->  (current)
```

All four ended with **no systemd shutdown sequence** — the journal simply stops
mid-operation (last entries are the 30 s `eduroam-watchdog` tick). For contrast,
the Jul 23 and Jul 27 shutdowns *do* appear as clean `shutdown` records in
`last -x`. So these four are crashes or power loss, not reboots.

Ruled out, across all four boots (`journalctl -b <n> -k`):
no I/O error, no ext4 error, no MCE / `Hardware Error`, no OOM-kill,
no NVIDIA `Xid`, no thermal throttle.

Load and memory at the crash (from `sysstat`, which is collecting):

| crash | ldavg-1 (24 CPU) | kbcommit | %commit |
|---|---|---|---|
| Jul 30 15:30 | 7.60 | 268 GB | **99.46** |
| Jul 30 20:01 | 5.06 | 167 GB | ~55 |
| Jul 31 04:39 | 11.50 | 192 GB | ~71 |
| Jul 31 07:25 | 9.74 | 147 GB | ~35 |

Swap unused throughout (249 GB RAM + 8 GB swap). So memory pressure is **not**
the common cause, though crash 1 alone was at the commit ceiling and may have a
different explanation from the other three.

One clue, 8 s before the crash-4 log ends:

```
Jul 31 07:25:45 hugin munin-paper-detect[5313]: [ERROR] ... [Errno 30] Read-only file system
```

A filesystem went read-only. That would also stop journald persisting, which is
consistent with a panic leaving no trace afterwards.

**Not root-caused.** The remaining suspects need hardware access this session did
not have (`smartctl` requires sudo and timed out). To check next:
`sudo smartctl -a /dev/sda` (the 22 TB ST24000NM002H holding `/mnt/hdd`) and the
root device; PSU adequacy under sustained GPU load; whether the machine is on a
switched/shared circuit.

### Plus a weekend power-down

Jul 31 07:25 → Aug 3 07:46: the machine was **off for 3 days** — 3 of the 4 days
since the sweep launched. Same class of event as the ab42 array 830 loss.

### And the auto-resume service is broken

```
× slurm-auto-resume.service - Auto-resume SLURM nodes after reboot
     Active: failed since Mon 2026-08-03 07:52:33
     bash[10981]: slurm_update error: Invalid node state specified
```

It fails on every boot, so the node does not return to service on its own.

### Conclusion

The crashes are an infrastructure problem we do not control and have not yet
diagnosed. Long single-shot jobs cannot be made reliable here. Runs must survive
requeue — that is the only lever available on our side.

---

## Part B — what resume machinery already exists

| piece | location | state |
|---|---|---|
| `--resume <exp_dir>` CLI flag | `args.py:271` | exists |
| `resume_experiment_directory()` | `master_pipeline.py:495` | exists, reuses dirs |
| `_discover_trained_models()` | `master_pipeline.py:461` | exists |
| `_discover_dataset_path()` | `master_pipeline.py:480` | exists |
| training-phase skip if model exists | `master_pipeline.py:170-179` | exists — **but unsound, see Gap 1** |
| `--skip_preparation` / `--skip_training` / `--only_analysis` | `args.py:273-277` | exist |

**The SLURM script never passes `--resume`.** Every requeue calls
`setup_experiment_directory()`, which mints `exp_gtt_<timestamp>` — a fresh dir,
fresh cache, training from epoch 0. That is the whole reason 11 attempts produced
nothing: the capability is there and unused.

But turning it on as-is would be **worse than the status quo**, because of Gap 1.

### Gap 1 — a half-trained model looks finished (correctness, blocking)

`best_model.pt` is written on *every* validation improvement, from epoch 1
(`vampnet/vampnet.py:994-998`). A run killed at epoch 61/100 leaves one behind.
On resume, `run_training_phase` (`master_pipeline.py:170-179`) sees it and
**skips the combo**, then proceeds to analysis and the JSD retrain loop using an
undertrained model, with nothing in the output recording that it is undertrained.

That silently turns a 61-epoch model into a published 100-epoch result. This must
be fixed before `--resume` is enabled anywhere.

### Gap 2 — no epoch-level resume

`checkpoint_epoch_N.pt` is written every `save_every` epochs
(`vampnet.py:1028`) but **nothing ever loads it**. And `save_complete_model` is
`torch.save(self)` (`vampnet.py:656`) — the model object only: no optimizer
state, no scheduler, no epoch counter, no RNG state. Even a manual resume would
restart the Adam moments and the data order.

### Gap 3 — analysis never skips completed work

`run_analysis_phase` (`master_pipeline.py:201`) re-runs analysis for every model
in `trained_models` unconditionally. Analysis is the dominant cost
(~3.5 h vs ~12 min training). A job that dies in retrain round 2 would redo every
earlier analysis on resume.

Related: the retrain loop keeps its round counter only in memory, so a resumed run
re-derives it from whatever models happen to exist on disk.

---

## Part C — the plan

Ordered by dependency. Steps 1–2 are the blocking correctness work; 4–5 are what
actually make requeue cheap; 3 is the deepest win and can follow.

> **Status 2026-08-03: steps 1–5 implemented, 15/15 tests in `tests/test_resume.py` green.**
> Step 6 (τ=50 stride bug) is still open. See "What changed" at the end.

### Step 1 — regression test first
Build a fixture exp dir containing a `best_model.pt` written at epoch 61 of a
100-epoch request, then assert the pipeline does **not** treat it as complete.
Run it, watch it fail, record the magnitude. (Standing preference: pin behavior
with a test before changing it.)

### Step 2 — completion markers *(fixes Gap 1 and Gap 3)*
- Write `training_complete.json` next to `best_model.pt` **only after the epoch
  loop finishes** — `{epochs_run, epochs_requested, best_score, config_hash, git_sha}`.
- Change the skip test at `master_pipeline.py:170-179` from "`best_model.pt`
  exists" to "marker exists **and** `epochs_run == epochs_requested`".
- Same for analysis: `analysis_complete.json` per analysis dir, checked at
  `master_pipeline.py:201`.

After this, `--resume` is safe and skips completed *phases*. An interrupted
training still restarts that model from epoch 0, but never gets mistaken for done.

### Step 3 — real checkpoint/resume in the trainer *(fixes Gap 2)*
Extend the periodic checkpoint to a dict
`{model_state, optimizer_state, scheduler_state, epoch, best_score, rng_states}`.
Keep `save_complete_model` as-is for the final artifact so nothing downstream
breaks. Add `--resume_training`: load the newest `checkpoint_epoch_N.pt`, restore,
continue at N+1. With `--save_every 10` a crash costs ≤10 epochs instead of the
whole run.

### Step 4 — stable experiment directory
Add `--exp_name` (or `--resume auto`) so each (seed, lag) gets a deterministic dir
— `exp_gtt_s0_lag10` — instead of a timestamp. Requeue then reuses it with no
discovery step, and the dataset cache inside it (0.5 GB at τ=5, 0.2 GB elsewhere;
the 8-11 GB figures earlier are whole attempt directories, not the cache) is reused
rather than rebuilt.

### Step 5 — SLURM script changes
- `#SBATCH --requeue` (explicit)
- `#SBATCH --open-mode=append` — **right now each requeue clobbers the `.out`**,
  which is why 5 attempts left only 1 `Start:` banner and why the crash evidence
  from earlier attempts is gone
- `--signal=B:USR1@120` + a trap that forces a checkpoint before the job dies
- pass `--resume auto --resume_training`

### Step 6 — separate issue, already scoped
τ=50 ns analysis-stride divisibility bug (Phase 3 collapses prep × runtime stride;
50 ns is not a multiple of the resulting 4 ns). Independent of resume.

### Infrastructure asks (need sudo / not ours)
- Fix `slurm-auto-resume.service` ("Invalid node state specified")
- `sudo smartctl -a /dev/sda` and root device
- Determine whether the weekend power-down is scheduled and can be avoided

---

## Open decisions

1. **Scope now**: Steps 1–2 + 4–5 (safe resume, phase-level reuse) as one unit,
   with Step 3 (epoch-level) as a follow-up? Or all of 1–5 in one pass?
2. **The two running jobs** (τ=5 attempt 5, τ=10 attempt 3) are writing into
   non-resumable timestamped dirs and will be lost at the next crash. Let them
   run, or cancel and relaunch once resume lands?

## What changed (implemented 2026-08-03)

| file | change |
|---|---|
| `pygv/vampnet/vampnet.py` | `_save_resume_state` / `_load_resume_state` (model + optimizer + scheduler + epoch + best/plateau bookkeeping + RNG, written atomically); `fit(resume_state_path=...)` starts at `start_epoch`; `history['epochs_run']` records what actually ran |
| `pygv/pipe/training.py` | `write_training_marker()` → `training_complete.json`, written **only after** the epoch loop returns; passes `resume_state_path` when `--resume_training` |
| `pygv/pipe/master_pipeline.py` | skip test is now marker-based (`_completed_model_path` / `_partial_model_path`); `--exp_name` pins the experiment dir; `--resume auto`; resumed training reuses the interrupted run's own timestamped dir; analysis skip + `analysis_complete.json`; skipped analyses restore their diagnostic verdict |
| `pygv/pipe/args.py`, `config/base_config.py` | `--exp_name`, `--resume_training`, `--save_every`; guard: `--resume_training` with `--save_every 0` exits with an explanation |
| `cluster_scripts/gtt_lagsweep_v1_array.sh` | `--requeue`, `--open-mode=append`, per-(seed,lag) `--exp_name`, `--resume_training --save_every 10`, restart-count in the banner, preflight check that the `pygvamp` on PATH supports the new flags |

Four traps found and closed while implementing. The first was caught by the full
test suite (`test_phase5_integration`); the rest would have made resume quietly
useless rather than loudly broken:

0. **Passing `resume_state_path` unconditionally into `model.fit()` killed every
   reversible run** — `RevVAMPNet` overrides `fit()` and does not accept it
   ("unexpected keyword argument"). Training failed, so no models reached analysis.
   Fixed: the kwarg is passed only when resuming, and requesting resume against a
   training loop that cannot honour it (RevVAMPNet, or the three-phase path) now
   raises with an explanation rather than silently dropping the request.

1. **Resumed training would have written to a fresh timestamped run dir**, so the
   previous `resume_state.pt` would never be found — resume silently degrading to
   a full restart. Fixed by pinning `run_name` to the interrupted run's directory.
2. **A skipped analysis returned a bare stub**, and `_run_retrain_loop` skips any
   experiment whose `diagnostic_report` is None — so a resumed run would have
   stopped reducing k, disabling the exact capability the GTT sweep measures.
   Fixed by persisting the verdict in the analysis marker and restoring it.
3. **`_discover_trained_models` still used the old "any best_model.pt" rule.**
   That is the path retrained models (`lag10ns_9states_retrained`) reach analysis
   by — `run_training_phase` never iterates over them. An interrupted retrain round
   would have been analysed and reported as a converged k, i.e. a wrong k*(τ)
   datapoint, which is the sweep's actual output. Now marker-aware like the rest.

### Deviations from the plan above (deliberate)

1. **Step 2's rule changed.** The plan said "marker exists AND
   `epochs_run == epochs_requested`". That is wrong: early stopping legitimately
   finishes with `epochs_run < epochs_requested`, and the plan's rule would have
   retrained every converged model forever. Implemented rule: the marker's
   presence proves the run was not interrupted (it is written only after the epoch
   loop returns), and the remaining check is whether it covers the budget being
   asked for now — `marker.epochs_requested >= config.epochs`. Pinned by
   `test_early_stopped_run_counts_as_complete` and
   `test_marker_from_a_smaller_epoch_budget_is_not_reused`.

2. **Step 5's `--signal=B:USR1@120` trap was NOT implemented.** It buys a graceful
   checkpoint window when SLURM terminates a job — scancel, wall-clock timeout, or
   preemption. None of those are the failure mode here: the partition has no time
   limit, preemption is off, and a hard crash or power loss gives no signal at all.
   The rolling `resume_state.pt` every `--save_every` epochs is what actually covers
   this failure mode. Worth adding if the node ever starts being preempted; not now.

### Deployment note — this WILL bite if missed
Jobs run the installed `pygvamp/1.0.0` module, not the working tree, and that
module has none of these flags. Until it is redeployed, launch with:

```
PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP sbatch --array=0-3%2 \
    cluster_scripts/gtt_lagsweep_v1_array.sh
```

The script now preflights `pygvamp --help` for `--exp_name` and exits with this
message rather than letting 27 tasks die on an argparse error.

## Measured facts to size against (do not estimate)
- τ=10 ns attempt 1 reached retrain round 2 (k: 10 → 9 → 3) in ~6.5 h before crash 1.
- τ=5 ns has never completed the first k=10 training in 5 attempts (~10 min/epoch,
  100 epochs, plus analysis, plus up to 5 retrain rounds).
- τ=50 ns is the only rung that ran to exit 0 — 2 h 45 m — and its analysis failed.
