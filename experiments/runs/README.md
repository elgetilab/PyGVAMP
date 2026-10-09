# Run outputs (summary bundle)

Mirror of `/mnt/hdd/experiments/` (same directory layout), reduced to what fits
in git. Built by `experiments/scripts/build_results_bundle.sh`. Re-run it to
refresh.

**Kept verbatim:** every `*.json`, `*.csv`, `*.yaml`, `*.txt` in each run dir:
`pipeline_summary.json`, `training_complete.json`, `analysis_complete.json`,
`discovery_summary.json`, state diagnostics, ITS summaries, transition
matrices, state counts, configs, metadata, `villin_tau_scan.csv`.

**Kept as digests:**
- `<run>/logs/log_*.digest.txt`: per-run training logs.
- `logs/*.out.digest`, `logs/*.err.digest`: SLURM job logs.

A digest drops tqdm progress bars, per-batch `Batch i/n, VAMP` lines,
mid-epoch `Quick validation` lines and blank lines. It also keeps only the
first copy of the `Lag time ... corresponds to` and `Loaded N trajectories`
lines, and collapses consecutive identical lines to one line suffixed
`[repeated N times]`. Per-epoch `Train VAMP / Val VAMP` lines, best scores,
warnings (including NaN warnings), errors and tracebacks are all kept. Older
runs predate `training_complete.json`, and for those the best VAMP-2 score
lives only in the training-log digest (`Loaded best model with score: ...`).

**Not included:** model checkpoints (`*.pt`), caches (`*.pkl`), arrays (`*.npz`,
`*.npy`), plots, HTML reports, PDB/PyMOL files, and the trajectories in
`/mnt/hdd/data/`. Paths inside configs and logs still point at `/mnt/hdd`.
