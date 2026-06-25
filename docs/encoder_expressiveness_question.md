# Why don't more expressive graph encoders improve VAMPNet state decomposition?

*A self-contained briefing for further discussion. No codebase context required.*

## TL;DR

We compare three graph encoders inside a VAMPNet that learns metastable states
from molecular-dynamics (MD) trajectories. Two of them (GIN, ML3) are
**more expressive in the Weisfeiler–Leman (WL) / graph-isomorphism sense** than
the third (SchNet), so the prior was that they should resolve more structure and
score higher. Empirically they **do not**: properly tuned, GIN only *ties*
SchNet and ML3 *underperforms* (and is unstable across seeds). The working
explanation is that **WL/topological expressiveness is the wrong axis for this
task** — the discriminative signal is continuous 3D *geometry*, which all three
encoders already ingest, so extra power to distinguish graph *topologies* has
little to bite on. What actually separates the encoders is **inductive bias and
trainability (conditioning), not isomorphism-distinguishing power.** This doc
lays out the setup, the evidence, and the open questions.

---

## 1. Background (self-contained)

**The pipeline.** A VAMPNet learns a soft assignment of MD frames to `k`
metastable states by maximizing the **VAMP-2 score** — a variational lower bound
whose maximizer recovers the leading singular functions of the Koopman/transfer
operator at a chosen lag time τ. Higher VAMP-2 = the network has captured more of
the system's slow dynamics. For `k` states the score is bounded above by `k`
(sum of `k` squared singular values, the first being the trivial stationary one).

**The graph.** Each MD frame is turned into a graph: nodes = Cα atoms, edges =
a **k-nearest-neighbor graph** (here k=7) in 3D space, with edge features =
a radial-basis (Gaussian RBF) expansion of the interatomic **distance**. A graph
encoder maps each frame-graph to a fixed-size embedding; a small classifier +
softmax produces the `k`-state probabilities; the VAMP-2 loss is computed on the
resulting probability trajectory.

**The encoders compared:**
- **SchNet** — continuous-filter convolutions. Each message is the neighbor's
  features multiplied by a *filter that is a smooth learned function of the
  interatomic distance*. It is a **geometric** GNN: its power lies in
  distance sensitivity. On the abstract-graph WL hierarchy it is *not*
  1-WL-complete.
- **GIN** — Graph Isomorphism Network. Provably as powerful as the **1-WL test**
  for distinguishing non-isomorphic graphs (Xu et al., 2019). Gets its power from
  **sum aggregation** + an MLP update. Our variant (GINE-style) does use the
  distance edge features: `message = act(x_j + edge_proj(edge_attr))`.
- **ML3** — a custom multi-layer message-passing encoder with optional
  **spectral** edge filters (learnable frequencies, a receptive-field parameter)
  and parallel attention. More expressive machinery than either above.

**Expressiveness, precisely.** The "1-WL" hierarchy measures the ability to
distinguish **non-isomorphic graph topologies**. GIN sits at the top of that
hierarchy among message-passing GNNs; SchNet does not. But there is a *separate*
axis — **geometric expressiveness** (the ability to distinguish 3D point clouds /
geometric graphs; cf. "geometric WL", Joshi et al., 2023) — on which
distance-based encoders like SchNet are specifically strong. **These two axes are
largely orthogonal**, and this distinction is the crux of the puzzle below.

---

## 2. The benchmark and the puzzle

- **System:** Trp-cage (DESRES 2JOF), a 20-residue mini-protein. Single long
  Cα-only trajectory, ~1.04M frames at 0.2 ns/frame.
- **Task:** VAMPNet, lag τ = 20 ns, k = 5 states, 10 random seeds, per-seed
  VAMP-2 reported as mean ± std across seeds.
- **Literature anchor:** ~4.79 ± 0.01 for this setup (Ghorbani et al., 2022).

**Prior:** GIN and ML3 are more WL/spectrally expressive than SchNet, so they
should distinguish more graph structure and score ≥ SchNet.

**Observation:** they don't. Details below.

---

## 3. Results (the evidence)

All runs share the benchmark-invariants (same data, lag, k, k-NN graph with k=7,
val split, 100 epochs, seed set). Two regimes were run:
- **De-tuned:** every encoder forced into SchNet's hyperparameters (width 16,
  **no batch-norm**, no embedding, etc.) — a clean single-variable encoder swap.
- **Native:** each encoder restored to its *own* preset architecture
  (its native width, **batch-norm**, embedding, learning rate), holding only the
  benchmark-invariants fixed — a best-vs-best comparison. (One forced deviation:
  GIN/ML3 native batch size of 32 is infeasible at 1.04M frames (~3 days/seed),
  so batch=1000 was used; batch-norm — not the tiny batch — is the hypothesized
  stabilizer.)

| Encoder | Regime | Seeds | VAMP-2 (mean ± std) | Notes |
|---------|--------|-------|---------------------|-------|
| **SchNet** | baseline | 10 | **4.6516 ± 0.0175** | tight spread |
| GIN | de-tuned | 10 | 4.5955 ± 0.0750 | worse + noisy |
| **GIN** | **native** | 10 | **4.6481 ± 0.0343** | **≈ ties SchNet**, variance halved |
| ML3 | de-tuned | 1 (smoke) | 4.6431 | single seed only |
| **ML3** | **native** | 10 | **4.5743 ± 0.0770** | **worse mean + high seed instability** (worst seeds ~4.44) |

Supporting sub-findings:
1. **De-tuning, not the encoder, caused GIN's gap.** Restoring batch-norm + width
   + embedding moved GIN from 4.5955 → 4.6481 (SchNet parity) and *halved* its
   cross-seed variance. The single most important restored ingredient is
   **batch-norm** — GIN's sum aggregation produces unnormalized magnitudes that
   it tames.
2. **GIN's deficit was underfitting, not overfitting.** In the de-tuned runs
   train-VAMP ≈ val-VAMP (both low) — the model converged to a *worse optimum*,
   it did not overfit. This points at optimization/conditioning, not capacity.
3. **ML3 does not recover** in its native regime: lower mean (4.5743) *and* high
   seed-to-seed instability (±0.077; worst seeds collapse to ~4.44). Its deficit
   looks **intrinsic** (the spectral/multi-layer machinery is harder to train
   reliably), not merely de-tuning.
4. **All three native encoders land at ~4.65**, ~0.14 below the paper's 4.79 —
   a **shared gap independent of encoder choice**.
5. **Implementation note (geometry handling):** the GIN variant L2-normalizes each
   edge's RBF distance vector before use (`edge_attr / ||edge_attr||`), partially
   discarding distance *magnitude*, whereas SchNet's filter preserves the full
   distance→weight mapping. So even the geometry GIN uses is slightly blunter.

---

## 4. Working hypothesis: the limiter is geometric/optimization, not isomorphism

**Claim:** WL/topological expressiveness is not what bounds performance here.

1. **The signal is geometric.** Distinct metastable states differ in their
   Cα–Cα **distance map** (continuous geometry / fold), not in the *topology* of a
   k-NN graph. A k=7 graph over 20 compact Cα atoms has limited topological
   variation across conformations — the informative differences live in the edge
   *distances*. WL power is about distinguishing topologies; it has little to
   exploit when the topology is near-constant and the geometry carries the signal.
2. **All three already see the geometry** (via the RBF distance edge features), so
   none is "blind" to the discriminative axis; they capture it comparably.
   SchNet's continuous-filter conv is simply the most *direct* and best-conditioned
   way to turn distance into a message.
3. **What differs is trainability, not capacity.** Sum aggregation (GIN) needs
   normalization; spectral layers (ML3) are sensitive to initialization. Restoring
   batch-norm fixes GIN; nothing in the native regime fixes ML3's seed variance.
4. **Therefore extra expressiveness on the topological axis neither helps (GIN
   ties) nor rescues a poorly-conditioned encoder (ML3 stays behind).** The
   ordering that emerges — SchNet ≈ GIN > ML3 — tracks *geometric inductive bias +
   conditioning*, not WL rank.

---

## 5. Caveats (what we have NOT shown)

- **One system, one lag, one k, one graph (k=7).** No generalization claim across
  proteins, lag times, or neighbor counts.
- **The val metric is a noisy quick-validation** (sampled every N batches);
  train-VAMP saturates near its theoretical max in all runs. We have not persisted
  a clean full-validation curve, so "underfit vs overfit" rests on a few points.
- **"ML3 is intrinsically unstable" is a hypothesis**, supported by cross-seed
  variance but not by a mechanistic ablation.
- **We never directly tested whether k-NN topology distinguishes the states** —
  the geometric-vs-topological claim is argued, not measured (see Q1 below).
- The ~0.14 gap to the paper is unexplained and may be unrelated to encoders
  (featurization, lag/τ-normalization, training length, etc.).

---

## 6. Open questions to delve into

1. **Is k-NN graph topology actually uninformative for these states?**
   *Measurable:* compute WL colorings / graph descriptors of frame-graphs and test
   whether they separate the known metastable states at all, vs the distance map.
   If topology barely separates states, that directly confirms the hypothesis.
2. **Does geometric expressiveness (not topological) predict a better encoder?**
   Would E(3)-equivariant / directional message passing (angles, relative
   orientations — features beyond pairwise distance) help where 1-WL power didn't?
   This is the "right axis" version of the original question.
3. **What is the shared ~0.14 gap to 4.79?** Architecture-independent, so suspect
   featurization, lag-time/τ handling, training schedule, or the VAMP-2 estimator —
   not the encoder. Worth separating from the encoder question.
4. **Why is ML3 unstable across seeds?** Init scheme for the spectral layers?
   Learning-rate/conditioning of the frequency parameters? Loss landscape?
5. **Is VAMP-2 train-saturation masking the real comparison?** Train-VAMP pins near
   the max (k) for every encoder; the discriminating signal is entirely in
   validation/dynamics quality (CK test, implied timescales). Should the comparison
   be made on those rather than on VAMP-2 alone?

---

## 7. Reference numbers (for quick recall)

| | VAMP-2 (k=5, τ=20 ns, Trp-cage) |
|---|---|
| Paper (Ghorbani 2022) | 4.79 ± 0.01 |
| SchNet (native) | 4.6516 ± 0.0175 |
| GIN (native) | 4.6481 ± 0.0343 |
| ML3 (native) | 4.5743 ± 0.0770 |
| GIN (de-tuned) | 4.5955 ± 0.0750 |

Theory anchors to look up: GIN ↔ 1-WL (Xu et al., *How Powerful are Graph Neural
Networks?*, 2019); geometric GNN expressiveness / geometric WL (Joshi et al.,
*On the Expressive Power of Geometric GNNs*, 2023); SchNet (Schütt et al., 2017);
VAMPNets (Mardt et al., 2018); the Trp-cage VAMPNet benchmark (Ghorbani et al.,
2022).
