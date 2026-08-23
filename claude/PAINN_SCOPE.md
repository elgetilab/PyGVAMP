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

> **Stronger correction, established during implementation (2026-08-21).**
> PaiNN's scalar readout is **also reflection-invariant**, so it does not capture
> chirality either. The vector channel couples back to scalars only through
> orthogonally-invariant contractions — norms ‖v‖ and dot products ⟨v, w⟩ — and
> ⟨Rv, Rw⟩ = ⟨v, w⟩ holds for improper R (det = −1). Detecting handedness would
> need an odd-order invariant such as a triple product v₁·(v₂×v₃), which standard
> PaiNN does not form. Pinned by
> `test_scalar_readout_is_reflection_invariant_documents_a_limitation`.
> **The chirality framing is therefore dead for PaiNN as well, not just for the
> angular pre-test.** PaiNN's only claim over a distance-only descriptor is
> angular resolution through directional message passing.

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
| D5 | Capacity matching vs SchNet | equal width, or equal parameter count | **MEASURED 2026-08-21** — see table below. Equal width is 2.1× SchNet's parameters on Trp-cage. Recommend parameter-matched |
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

---

## Implementation record (2026-08-21)

Built: `pygv/encoder/painn.py`, `tests/test_painn_encoder.py` (17 tests),
`PaiNNConfig`, `--model painn` + six `--painn_*` flags, `pos` on the graph,
`requires_pos` plumbing in VAMPNet and RevVAMPNet.

**The blocking prerequisite cost nothing, as predicted.** The cache stores raw
frames and graphs are built per `__getitem__`, so adding `pos` invalidated no
existing cache (Trp-cage, Villin, NTL9, GTT all still valid).

### D5 resolved with measurements, not estimates

Encoder-only parameter counts, `edge_dim=16, output_dim=16, n_interactions=4`:

| | SchNet | PaiNN (equal width 16) | ratio |
|---|---|---|---|
| node_dim=20 (Trp-cage) | 7,600 | 15,952 | **2.10×** |
| node_dim=35 (Villin/GTT) | 14,200 | 16,192 | 1.14× |

The ratio is strongly system-dependent, because SchNet's parameter count scales
with `node_dim` (one-hot node features = n_atoms wide) while PaiNN's is dominated
by `hidden_dim`. **Parameter-matched setting for Trp-cage: `--painn_hidden_dim 10`
→ 7,276 params (0.96× SchNet).** Recompute per system; do not reuse this number.

An earlier note here guessed "~4× per-node state". Wrong on the parameter question
— per-node *state* is ~4×, per-*parameter* count is 2.1× at node_dim=20 and only
1.14× at node_dim=35. Use the measurement.

### Two corrections established while building

1. **Chirality framing is dead for PaiNN too** (see §1 blockquote). Standard PaiNN
   with a scalar readout is reflection-invariant. Angular resolution is the only
   claim.
2. **§4 was a real defect, not a hypothetical.** The first end-to-end run exited 0
   with `analysis_completed: []` and 11 stray files: no attention → the atom-count
   inference in `calculate_state_edge_attention_maps` raised → Phase 3 swallowed
   it. Fixed with an announced skip; the same run now yields
   `analysis_completed: ['lag20.0ns_5states']` and 77 artifacts. Option (a) from
   §4, as recommended.

### Still open
- `knn_angular_features` has a per-centre Python loop (fine at 20 atoms, would hurt
  at NTL9 scale). Only matters if the angular features are used in a production arm.
- PaiNN runs leave an empty `edge_attentions/` directory. Harmless; no misleading
  files are written.
- **Not yet benchmarked.** No PaiNN number exists on any system.

---

## k-NN truncation measurement (2026-08-23) — does NOT support the size argument

Ran `cluster_scripts/measure_knn_truncation.py` over four controls and three
candidates, 3000 frames each, k as used in production.

| system | atoms | retention | hop_cov | unexpl_var | state_lev |
|---|---|---|---|---|---|
| trpcage *(angular test NULL here)* | 20 | 0.446 | 1.000 | **0.719** | 0.768 |
| villin | 35 | 0.345 | 0.994 | 0.628 | 0.896 |
| gtt | 35 | 0.351 | 0.991 | 0.730 | 0.893 |
| ntl9 | 39 | 0.309 | 0.995 | 0.482 | 0.798 |
| a3d | 73 | 0.160 | 0.946 | 0.452 | 0.726 |
| nug2 | 56 | 0.211 | 1.000 | 0.058 | 0.781 |
| lambda | 80 | 0.148 | 0.967 | 0.743 | 0.729 |

### The size argument was half right and it was the wrong half
Retention falls with system size exactly as predicted (0.45 → 0.15). But retention
is not what matters — what matters is whether the *discarded* pairs carry
information the retained ones lack, and that does **not** track size:
α3D (0.45) and NuG2 (0.06) are BELOW every control; λ (0.74) is level with
Trp-cage (0.72) and GTT (0.73). Bigger proteins are more geometrically
over-determined, so knowing 16% of the distances pins much of the rest.

### The decisive calibration
**Trp-cage sits at 0.72 unexplained variance and the angular pre-test was still
null there.** So a high value does not predict any angular benefit, and no
candidate exceeds Trp-cage by enough to matter. On the information axis, none of
these systems is better motivated for PaiNN than the one already tested.

### Hop coverage kills the weaker version of the argument too
At `n_interactions=4`, message passing already reaches **95–100%** of all pairs on
every system. A pair missing from the k-NN graph is not invisible to the network.
The strongest form of "the truncation hides geometry" therefore does not hold.

### An artifact that inverted the answer — caught before reporting
The first pass fitted the retained→discarded regression **in-sample**. On α3D (423
predictors vs 400 frames) and λ (467 vs 400) the fit is perfect by construction, so
it reported unexplained variance ≈ 0.006 and 0.038 — i.e. "the big proteins discard
nothing", the exact opposite of the corrected result. Fixed with a 70/30 held-out
split, ridge chosen on the hold-out, 3000 frames, plus a `predictors_ge_samples`
flag so the condition cannot recur silently.

### What this does and does NOT establish
It bounds the **information** argument only: the discarded geometry is largely
recoverable, and the receptive field already covers it. It does **not** bound an
**inductive-bias** argument — explicit directional features might still make the
same information easier to *learn*. Neither this measurement nor the angular
pre-test tests that. A PaiNN run remains legitimate as a capability demonstration
or an optimisation question; it is just no longer supported as an information one.

### Recommendation
Do not pick a target using the truncation argument — it does not discriminate.
If PaiNN runs for completeness, choose on the system's own scientific interest:
**α3D** (73 CA, three-helix bundle, misfolding-rich landscape, plausible k*>2) or
**λ-repressor** (80 CA, five helices, richest landscape, most expensive).
**NuG2 is the worst candidate** (unexpl_var 0.058 — β-sheet rigidity makes its
long-range geometry almost fully determined by local contacts).
