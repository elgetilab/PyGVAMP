# AT1R — d125d — Experiment Tracker

## System Info

| Property | Value |
|----------|-------|
| Protein | Angiotensin II receptor type 1 (AT1R) + 7-mer peptide |
| Protonation state | **d125d** (the `d` variant; `d125p` deferred) |
| Topology | `/mnt/hdd/data/julia_ATR/gmm0/r1/d125d/prot_chains.pdb` |
| Trajectories | `/mnt/hdd/data/julia_ATR` (recursive, pattern `d125d/cutted_dt_1ns.xtc`) |
| Trajectory count | 40 (gmm0–7 × r1–5, d125d only) |
| Total atoms | 5446 (protein + peptide; no solvent/membrane) |
| Chains | A = receptor (~319 res), B = 7-mer peptide |
| Selection | `chainid 0 and name CA` — **receptor CA only; peptide (chain 1) excluded** |
| Timestep | 1 ns/frame (`cutted_dt_1ns.xtc`) → lag in ns directly |
| Preset | `medium_schnet` (was `large_schnet` for run 0 / job 657 — switched down after that run overfit) |
| n_neighbors | **10** (was 20 for run 0; reduced for sparser, more local graphs + less memory) |

> The peptide (chain B / `chainid 1`) is deliberately excluded. The exact
> receptor CA count (~319) is printed by the discovery job as `Selected N atoms`.

## State Discovery

| Property | Value |
|----------|-------|
| Date | 2026-06-16 |
| Job ID | 655 |
| Selected atoms | 319 receptor CA (`chainid 0 and name CA`; peptide excluded ✓) |
| **Recommended n_states** | **10** (max across metrics; KMeans+GMM sweep over UMAP/t-SNE) |
| Output | `/mnt/hdd/experiments/atr_d/discovery/exp_atr_d_20260616_131749/` |

### Discovery command
```bash
sbatch cluster_scripts/atr_d_discovery.sh

# then, from the log:
grep "Recommended n_states" /mnt/hdd/experiments/logs/disc_<jobid>.out
grep "Selected"             /mnt/hdd/experiments/logs/disc_<jobid>.out
```

## Experiments

### Standard (VAMP-2)

| Run | Lag (ns) | Encoder | n_states | Stride | Epochs | Batch | Train VAMP | Val VAMP | Status | Job ID | Notes |
|-----|----------|--------------|----------|--------|--------|-------|------------|----------|--------|--------|-------|
| 0 | 20 | large_schnet | 4 | 5 | 50 | 256 | ~3.95 | ~3.4 (peak quick-val) | done | 657 | first run: n_states 4 (disc. 655 rec. 10), stride 5. Batch 2048 OOM'd (656) → 256. ~6 min. **Overfit** (train pinned ~4.0, val noisy/declining); diagnostic eigval-gap suggested 3, state 2 only 5% pop — but keeping 4 per decision |
| 1 | 20 | medium_schnet | 4 | 5 | 50 | 256 | — | — | pending (module rebuild) | — | medium_schnet + nn 10 (was large/nn20) to curb overfit; stride 5 still (test). Run after module rebuild |

#### Submit command
```bash
# Run 1 (medium_schnet, nn 10, stride 5): rebuild the pygvamp module FIRST,
# then submit. --run 1 keeps outputs separate from run 0.
sbatch cluster_scripts/atr_d_experiment.sh --n_states 4 --run 1
```

### Reversible (RevGraphVAMP)

Not planned at the moment (standard VAMP-2 only).

## Output Location

`/mnt/hdd/experiments/atr_d/discovery`        (state discovery)
`/mnt/hdd/experiments/atr_d_std/lag20/run_00` (standard run 0)

## Run 0 results (job 657)

- **Output:** `/mnt/hdd/experiments/atr_d_std/lag20/run_00/exp_atr_d_20260616_141014/`
- **Model:** `…/training/lag20.0ns_4states/20260616_141027/models/best_model.pt`
- **VAMP-2:** train ≈ 3.95–4.00 (max = 4 for 4 states); quick-val peaked ~3.38 → train/val gap (expected for a quick stride-5 `large_schnet` run).
- **Implied timescales** (3 non-trivial processes, mean across lags): ~1.5 µs, ~0.84 µs, ~0.30 µs.
- **Caveat:** absolute ITS/CK timescales are **not time-calibrated** — the post-training analysis loader subsamples non-contiguously. Treat µs values as relative; populations/qualitative separation are reliable.
- Analysis artifacts: ITS plot, CK test, state network, `atr_d_interactive_report.html`.

## Notes

- **Resources:** `gputraining` partition, `--gres=gpu:batch:1` (full 5090), `--cpus-per-task=16`, `--mem=128G`. (128G and 16 CPUs are the partition maxima: `MaxMemPerNode=128000`, `MaxCPUsPerNode=16`. 200G was requested but the partition rejects it with `MaxMemPerLimit`.)
- **n_states:** discovery-first workflow — run `atr_d_discovery.sh`, read the recommended value, then submit `atr_d_experiment.sh --n_states <N>`. No number is hardcoded.
- **epochs/batch:** 50 / 256 in `atr_d_experiment.sh` (batch reduced from 2048 after the run-0 OOM); edit the script to change.
- **Interactive report fix (`pygv/utils/interactive_report.py`):** the structure viewer now mirrors the **mdtraj training selection** (`training_selection`) instead of the full-protein `"protein"` viz-selection. With `chainid 0 and name CA` it embeds 319 CA atoms instead of 5434 → ~17× smaller (run-0 report was 1.88 GB / unloadable, driven by 4040 frames × 5434 atoms × xyz inline). Full-atom selections (e.g. small molecules) still show all atoms. Also bounded the structure-coordinate frame count to the same `max_frames` (5000) cap the analysis/embedding data already used (was `max_frames * 10` = 50000, so it embedded every frame for any dataset < 50k) — at stride 1 (~20k frames) it now strides down to ~5000 structure frames (~60 MB) instead of ~250 MB. **Requires the cluster module rebuild to take effect.**
- **⚠️ Module rebuild pending:** the code fix lives in this repo, but the cluster runs the installed `pygvamp/1.0.0` module. Rebuild it before submitting run 1 (and before regenerating any report).
- **d125p:** deferred. To add later, clone the two scripts and swap the file pattern to `d125p/cutted_dt_1ns.xtc` (topology `.../d125p/prot_chains.pdb`).
- **Legacy `cluster_scripts/atr.sh`:** superseded — points at a different cluster (`paula`, conda `PyGVAMP5`), uses the legacy `run_training.py`, and `name CA` (would include the peptide). Not reused here; retire later.