# PaiNN encoder — scope (2026-08-12)

Scoping only. No code written. Decisions to be made before any implementation.

Reference to verify before citing: Schütt, Unke & Gastegger, *Equivariant message
passing for the prediction of tensorial properties and molecular spectra*, ICML 2021.

---

## 1. Why PaiNN is a different bet from GIN/ML3

GIN and ML3 both changed the **aggregation** over the same inputs — the Cα pairwise
distance map, Gaussian-expanded. Neither beat SchNet on any of the three systems
(Trp-cage tie, Villin −0.10, NTL9 unstable/OOM). The recorded reading was that the
states are geometric and SchNet's RBF bias already captures them.

That result is better stated as: **the binding constraint was the information, not
the expressiveness.** Adding WL/spectral power over a fixed two-body descriptor
cannot recover information the descriptor never carried.

PaiNN is the first candidate that changes the **inputs** rather than the
aggregation. It maintains an equivariant vector channel built from the unit
displacement vectors r_ij/|r_ij|, so it carries *directional/angular* structure
that a distance-only representation does not contain.

This makes the arm falsifiable in a way a fourth aggregation variant would not be:

> **H:** if PaiNN ≈ SchNet, the Cα distance map is information-sufficient for VAMP
> state assignment on these systems, and the encoder lane can be closed for good.
> If PaiNN > SchNet, angular information matters and the whole descriptor choice
> (not the network) is the lever.

Either outcome is publishable and terminal. That is the argument for spending
cluster time here where GIN/ML3 no longer justify it.

**Honest caveat on the chirality framing.** A full pairwise distance matrix
determines a point set only up to rigid motion *including reflection*, so
distance-only encoders are blind to handedness. True — but MD frames never contain
mirrored structures, so this blindness likely costs nothing in practice. Do **not**
lead with chirality; the real claim is angular resolution, which is weaker but
honest.

---

## 2. Blocking prerequisite — the graphs carry no direction

`vampnet_dataset.py:455-490` computes the displacement vectors and immediately
throws the direction away:

```
diff = coords.unsqueeze(1) - coords.unsqueeze(0)     # displacement vectors
distances = torch.sqrt((diff ** 2).sum(dim=2))       # -> magnitude only
...
edge_attr = self._compute_gaussian_expanded_distances(edge_distances)
graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, num_nodes=...)
```

There is no `pos` and no `edge_vec` on the `Data` object. **PaiNN cannot be built
as a drop-in encoder against the current graph schema.**

### The good news: this is cheap

The cache stores **frames (coordinates)**, not prebuilt graphs — `_save_to_cache`
persists `'frames'`, and `__getitem__` calls `_create_graph_from_frame(idx)` per
access. So coordinates are already present at graph-build time.

Consequences:
- Adding `pos` to `Data` is a **graph-build change only**.
- **Every existing cache stays valid** — no re-prep of Trp-cage, Villin, NTL9 or GTT.
- Storage cost is `n_atoms × 3` per frame, far smaller than the existing
  `n_edges × gaussian_expansion_dim` edge features.

### Latent risk worth fixing while we are here
`_get_cache_filename` keys on `hash(traj_files) + lag + n_neighbors + stride +
cont_flag` — **no schema version**. Nothing in this change trips it (the cached
payload is unchanged), but any future change to *what* is cached would silently
load a stale, wrong-shaped cache. Add a `schema_version` to the key or the payload.

---

## 3. Design decisions to make (no code yet)

| # | Decision | Options | Lean |
|---|---|---|---|
| D1 | How direction reaches the encoder | `pos` on `Data` (n×3), or precomputed `edge_vec` (e×3) | `pos` — smaller, conventional, encoder derives r_ij from `edge_index` |
| D2 | Invariance boundary | readout from scalar channel only, vs any vector use | Scalar-only readout. The VAMP states **must** be rotation-invariant; PaiNN gives this for free if the readout ignores the vector channel |
| D3 | Frame alignment | require superposed frames, or not | Not required — equivariance is precisely what removes the need. Confirm current frames are unaligned so SchNet's invariance was doing the work |
| D4 | Node features | reuse learned embeddings / amino-acid encodings | Reuse. PaiNN conventionally takes species embeddings; the existing path is compatible |
| D5 | Capacity matching vs SchNet | equal width, or equal parameter count | **Decide before running.** PaiNN carries scalar+vector state per node (~4× the per-node state at equal width), so equal-width is not a fair comparison. Recommend parameter-matched |
| D6 | Which regime | de-tuned (single-variable swap vs paper-SchNet) or native preset | **De-tuned.** The known weakness of the existing table is that it was whole-recipe vs paper-SchNet, not encoder-in-isolation. A matched swap makes the claim clean |

---

## 4. Integration gap — PaiNN has no attention

Attention is a first-class analysis output: `analysis.py` steps 7–8 compute and
plot state edge-attention maps, and the run directories carry
`*_state_N_attention.png`, `*_state_attention_maps.npy`, residue-attention plots
and PyMOL scripts. All four working encoders expose `_attention_weights` /
`use_attention` (`schnet.py`, `gin.py`, `ml3.py`, `meta_att.py`).

**PaiNN has no attention mechanism by construction.** So either:
- (a) the analysis path needs a graceful skip when the encoder exposes no attention
  — the honest option, and it should *say* it skipped rather than emit nothing; or
- (b) a surrogate is invented (e.g. filter magnitudes as pseudo-attention) — this
  would be a non-standard modification and must not be presented as attention.

(a) is preferred. Note this makes PaiNN runs produce a strictly smaller artifact
set, which matters if the comparison figures assume attention maps exist.

---

## 5. Experiment protocol (Category 2)

Per `EXPERIMENT_CHECKLIST.md` Category 2: 10 seeds per (system, encoder), matched
architecture, fixed k and lag from the baseline paper, no discovery, no auto-stride.

- Systems: **Trp-cage** (τ=20 ns, k=5), **Villin** (τ=20 ns, k=4), **NTL9** (τ=200 ns, k=5)
- SchNet arm **already exists** — the Category 1 reproduction runs serve as baseline, no re-run
- PaiNN arm: **3 systems × 10 seeds = 30 runs**
- Reporting: mean VAMP-2 ± 95% CI per encoder; a "PaiNN beats SchNet" claim requires
  non-overlapping CIs (existing standard, carried over)

### Do not size this from the checklist's old "15 min/run"
That figure predates the measured cost model (analysis dominates; ~3.5 h analysis
vs ~12 min training, scaling with k). **Run one full probe on Trp-cage first**
(smallest system: 20 CA, 1.04M frames), read the wall time, then size the campaign.
Three wrong estimates on GTT is the precedent.

---

## 6. Risks, ranked

1. **NTL9 will be the hard one again.** GIN went unstable there (2/10 seeds collapse
   to VAMP=1.0) and ML3 CUDA-OOM'd 9/10 at shard:2. PaiNN is *heavier* than SchNet
   per node (vector channel), so budget a larger shard and expect this rung to be
   the one that fights back. Consider running NTL9 last.
2. **Capacity matching (D5) can invalidate the comparison** if decided post hoc.
   Fix it before the first run.
3. **No attention artifacts** (§4) — figures and downstream analysis that assume
   they exist will need a fallback.
4. **Equivariance bugs are silent.** A broken vector channel degrades accuracy
   without erroring. Needs a unit test asserting rotation-equivariance of the vector
   channel and rotation-*invariance* of the scalar readout, before any production run.

---

## 7. Cheaper falsification to consider first

Before building PaiNN, the hypothesis in §1 can be probed more cheaply: add an
angular/three-body feature to the *existing* SchNet path and see whether it moves
VAMP-2 at all on one system. If explicit angular information does nothing there,
PaiNN's directional channel is unlikely to, and the 30-run campaign can be
skipped — closing the encoder lane on a much smaller budget.

This is weaker evidence than PaiNN itself (a hand-made angular feature is not an
equivariant vector channel), but it is a legitimate cheap pre-test and worth
considering before committing.

---

## 8. Open questions for the user

- D5 and D6 above — capacity matching and regime — need deciding before any run.
- Is the PaiNN arm intended as a **methodological contribution** (PyGVAMP supports
  equivariant encoders) or a **scientific claim** (angular information matters for
  VAMP states)? The former justifies 1 system; the latter needs all 3.
- Should §7's cheap pre-test run first, or go straight to PaiNN?
