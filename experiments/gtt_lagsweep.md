# GTT / FiP35 WW domain — k*(τ) lag ladder — Tracker

Category 3 (framework capability demonstration, not a reproduction). Measures how
the recommended number of states k\* and the slow timescale t₂ vary across ~2 orders
of magnitude in lag time, using the JSD retrain loop rather than a clustering
heuristic.

## System Info

| Property | Value |
|----------|-------|
| Protein | WW domain FiP35 (DESRES GTT) |
| Data | `/mnt/hdd/data/gtt/` — GTT-0 (25 dcd) + GTT-1 (33 dcd) = 58 chunks |
| Selection | `name CA` — **35 atoms**, sequence `GSKLPPGWEKRMSRDGRVYYFNHITGTTQFERPSG` |
| Frames | ~5.6M (~1130 µs aggregate), 200 ps/frame |
| Timestep | **`--timestep 0.2` MANDATORY** (DESRES DCD metadata says 1 ps → 200× error) |
| Runs | GTT-0 and GTT-1 are **separate simulations**, never concatenated |
| Seeds | 1 (seed 0) — see caveats |
| Script | `cluster_scripts/gtt_lagsweep_v1_array.sh` |

Topology built by `cluster_scripts/mae_to_topol_pdb.py`, validated by regenerating
Trp-cage's hand-built topology byte-for-byte.

## Result: k\* = 2 across the entire ladder

Every rung converged on a genuine verdict (`recommendation=keep`,
`confidence=high`, no cap exhaustion):

| τ (ns) | descent | rounds | VAMP-2 @k=10 | VAMP-2 @k=2 | k\* | populations |
|---|---|---|---|---|---|---|
| 1 | 10→9→8→7→3→2 | 5 | 9.6652 | 1.9995 | 2 | 0.228 / 0.772 |
| 2 | 10→3→2 | 2 | 9.5801 | 1.9994 | 2 | 0.228 / 0.772 |
| 5 | 10→3→2 | 2 | 9.3455 | 1.9990 | 2 | 0.23 / 0.77 |
| 10 | 10→3→2 | 2 | 8.8415 | 1.9982 | 2 | 0.228 / 0.772 |
| 20 | 10→9→3→2 | 3 | 8.2231 | 1.9969 | 2 | 0.229 / 0.771 |
| 50 | 10→8→7→4→3→2 | 5 | 6.3599 | 1.9921 | 2 | 0.227 / 0.773 |

**k\*(τ) is flat at 2 over a 50× range in τ**, with populations identical to ~3
decimals. On this diagnostic FiP35 is two-state throughout — the textbook result.
VAMP-2 at k=10 declines monotonically with τ; at k=2 it sits at ~1.999 of a
ceiling of 2.

## The more interesting result: t₂ converges around τ ≈ 20 ns

Derived from the **validation VAMP-2** (for k=2, σ₂ = √(VAMP-2 − 1), t₂ = −τ/ln σ₂),
*not* from the pipeline's ITS output — the ITS/CK path decimates non-contiguously
so its absolute timescales are not time-calibrated, whereas VAMP-2 is estimated on
properly lagged pairs.

| τ (ns) | σ₂ | t₂ |
|---|---|---|
| 1 | 0.99975 | 4.00 µs |
| 2 | 0.99970 | 6.66 µs |
| 5 | 0.99950 | 9.99 µs |
| 10 | 0.99910 | 11.10 µs |
| 20 | 0.99845 | **12.88 µs** |
| 50 | 0.99604 | **12.61 µs** |

t₂ climbs and then **plateaus between τ=20 and 50 ns, agreeing to ~2%** — the
classic implied-timescale convergence test. The model is non-Markovian at short lag
and reaches Markovian behaviour around **τ ≈ 20 ns**, with a converged slow
timescale of **~12.6–12.9 µs**. Loosely corroborating: job 877's τ=50 ITS reported
max ~1430 ns, which scaled by its ~11× decimation lands near 15 µs — same ballpark.

*Check this against the specific FiP35 reference you intend to cite; it was not
validated against literature here.*

## ⚠️ Three separate caps biased k\* before the ladder was trustworthy

This is the main methodological lesson and it generalises beyond GTT.

1. **`--max_states` from above** (commit 6392da1). Discovery's max-across-metrics
   rule pinned `recommended_n_states` at the cap. Raising 10→25 did not fix it —
   BIC/AIC are monotone in k and simply re-pinned at 25 — and because
   `recommended_n_states` also feeds the *trained* model, every run trained k=24/25
   on a 35-residue protein. Resolution: start at k=10 and let the retrain loop
   descend; discovery's number carries no information here.
2. **A hard floor of 2 from below.** `pygv/utils/state_diagnostics.py:303` —
   `effective_n_states = max(2, min(effective_n_states, n_states))`. Once the loop
   reaches k=2 the recommendation is *necessarily* 2, so the convergence test fires
   by construction. Still defensible here because the `keep` verdict is independent
   of the clamped value: at k=2 every rung reported no underpopulated states and no
   JSD-mergeable pairs. And k=1 has no dynamics.
3. **`--max_retrains` truncating mid-descent.** τ=1 exhausted a cap of 5 at
   `10→9→8→7→6→3` while the diagnostic still asked for k=2, so k\*(τ=1) was never
   measured — 3 was just where the budget ran out. Rerun at cap 10
   (`GTT_MAX_RETRAINS=10 GTT_RUN_TAG=_r10`, job 896) converged to 2 in 5 rounds.

**Always check a k\* against the cap that produced it.** The loop warns on
exhaustion — `grep 'retrain loop exhausted'`.

## ⚠️ The descent path is not reproducible under a fixed seed

Same seed 0, same config except `max_retrains`, τ=1 descended two different ways:
`10→9→8→7→6→3` (capped run) vs `10→9→8→7→3→2` (rerun). VAMP-2 at k=10 also differed
(9.6676 vs 9.6652). Training is nondeterministic in practice, so the diagnostic's
median-of-three estimate tips differently between runs.

**The endpoint k\* was stable; the path was not.** Do not report descent paths as
findings. Error bars on k\* would need multiple seeds — this ladder is seed 0 only.

## Lag ladder construction (divisibility)

Every rung must be an integer multiple of `frame_dt × stride`, or the dataset
builder rejects it in ~2 s (`Invalid lag times: 1.0 ns -> closest valid: 0.0 ns`).
At 200 ps/frame:

| τ | stride | eff dt | lag in frames | cache frames |
|---|---|---|---|---|
| 1 ns | 5 | 1.0 ns | 1 | 1,137,344 |
| 2 ns | 10 | 2.0 ns | 1 | 568,672 |
| 5 ns | 5 | 1.0 ns | 5 | 1,137,344 |
| 10–50 ns | 10 | 2.0 ns | 5–25 | 568,672 |

**τ=0.5 ns is not achievable** — 2.5 frames at 200 ps, and stride only coarsens the
grid, never refines it. Nearest achievable neighbours are 0.4 ns (2 frames, stride
≤2) and 0.6 ns (3 frames, stride ≤3); both were held and are likely unnecessary
since k\* never moves off 2 down to τ=1.

Ladder capped at 500 ns on **training-set size**, not cost: auto-stride targets ~10
frames per lag, so at 1000–2000 ns only ~4–8k samples survive the 70/30 split —
too thin for a 35-CA net, and exactly at the end of the curve one would read.

## Costs (measured)

~285 min per (model × million training pairs). τ=10 was 8h23m/3 models at 568k
pairs; τ=5 16h28m/3 models at 1.14M; τ=20 10h26m/4 models at 568k; τ=1 27h/6
models. **Model count is not fixed** — it is however many retrain rounds fire — so
treat any per-lag estimate as a lower bound.

## Reproduce

```bash
# ladder is (seed, lag) grid; indices 0-3 = 5/10/20/50 ns, 4-5 = 1/2 ns (APPEND-ONLY)
sbatch --array=0-5%2 cluster_scripts/gtt_lagsweep_v1_array.sh
# raise the retrain cap and tag the output so a rerun stays distinguishable:
GTT_MAX_RETRAINS=10 GTT_RUN_TAG=_r10 sbatch --array=4 cluster_scripts/gtt_lagsweep_v1_array.sh
```

Array index maps to lag **by position** — new lags must be appended, never sorted
in, or the index→lag mapping of completed jobs changes retroactively.

## Open threads

- **Seed replication.** k\* = 2 is an integer and constant, so replication adds
  little there. But if **t₂ ≈ 12.7 µs** is to be published as a number it deserves
  uncertainty — 2–3 seeds at τ=20 and 50 only (the converged plateau), ~20–25h,
  rather than the whole ladder.
- 0.4 / 0.6 ns short rungs — held, likely unnecessary.
- ITS/CK absolute timescales remain uncalibrated (non-contiguous subsample); this
  is why t₂ above is derived from VAMP-2 instead.
