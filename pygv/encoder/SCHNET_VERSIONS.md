# SchNet encoder versions

PyGVAMP ships two SchNet variants.  Selection is controlled at the pipeline
level via `--encoder_variant {v1,v2}`.

## v1 — default (`pygv/encoder/schnet.py:SchNetEncoderNoEmbed`)

Original implementation.  Forward:

```
h = x
for interaction in self.interactions:
    delta, _ = interaction(h, edge_index, edge_attr)
    h = h + delta                  # residual
pooled = global_mean_pool(h, batch)
output = self.output_network(pooled)
```

## v2 — paper-faithful (`pygv/encoder/schnet_v2.py:SchNetEncoderNoEmbedV2`)

One-line difference: a per-atom `nn.ReLU()` is applied between the residual
loop and the global pool, mirroring the Ghorbani 2022 reference
(`github.com/ghorbanimahdi73/GraphVampNet`, `src/model.py:337` — variable
named `conv_activation`).

```
h = x
for interaction in self.interactions:
    delta, _ = interaction(h, edge_index, edge_attr)
    h = h + delta
h = self.post_conv_activation(h)   # <-- nn.ReLU(), per-atom
pooled = global_mean_pool(h, batch)
output = self.output_network(pooled)
```

The ReLU is hard-coded (not driven by `--activation`) because the reference
specifies `nn.ReLU()` regardless of the activation used inside the
InteractionBlock and CFConv.

## Why the order matters

ReLU does not commute with mean-pool.  Applying it per-atom before pool
gates negative per-atom activations independently; applying it after pool
only thresholds the aggregate.  This produces a different aggregate
representation, and in the Ghorbani 2022 architecture it appears to be
load-bearing — see `claude/VILLIN_REPRO_V5_LOG.md` for the empirical probe.

## When to pick which

- **v1** — backwards compatibility with all pre-v5 PyGVAMP runs.  Default.
- **v2** — strict reproduction of GraphVAMPNet (Ghorbani 2022) and follow-on
  work that uses the same encoder shape.

---

# Other encoders — quick status (2026-08-21)

| encoder | `--model` | attention | needs `pos` | status |
|---|---|---|---|---|
| SchNet v1/v2 | `schnet` | yes | no | baseline; best on every system tested |
| GIN | `gin` | yes | no | swept, closed — ties SchNet on Trp-cage, below on Villin, unstable on NTL9 |
| ML3 | `ml3` | yes | no | swept, closed — trails everywhere, least stable |
| **PaiNN** | `painn` | **no** | **yes** | implemented 2026-08-21, not yet benchmarked |
| Meta / MetaAtt | `meta` | yes | no | **BROKEN at runtime** — selectable but raises; see below |

## PaiNN (`pygv/encoder/painn.py`)

First encoder here that changes the *inputs* rather than the aggregation: it keeps
an equivariant vector channel built from the unit displacement vectors r̂_ij, so it
carries directional structure the Cα distance map discards.

* Needs node positions. `PaiNNEncoder.requires_pos = True`; VAMPNet/RevVAMPNet pass
  `data.pos` only to encoders that declare it. A graph without `pos` raises — it
  must never silently fall back to a distance-only path.
* **No attention.** The analysis phase detects this and skips attention artifacts
  with a `SKIPPED:` line; ITS/CK, transition matrices, populations and state
  structures are unaffected. PaiNN runs therefore produce a smaller artifact set.
* **Does NOT capture chirality.** The vector channel couples back to scalars only
  through orthogonally-invariant contractions (‖v‖, ⟨v,w⟩), which are unchanged by
  improper rotations. Its claim is *angular resolution*, nothing more.
* **`--painn_cutoff` defaults to none.** The paper uses a cosine cutoff, but these
  graphs are k-NN: a hard cutoff would drop neighbours the k-NN build kept.
* **Capacity is not comparable at equal width.** PaiNN stores 1 scalar + 3 vector
  components per channel per node. Match parameter counts explicitly in any arm.

Pinned by `tests/test_painn_encoder.py` (17 tests), including rotation
equivariance of the vector channel and rotation invariance of the readout — both
fail silently in production if broken.

## Meta / MetaAtt — broken, still selectable

`--model meta` reaches `create_model` but raises on the first forward:
`meta.py` falls back to `pygv/utils/alternative_torch_scatter` when `torch_scatter`
is absent (it is), but imports only `scatter_mean` while line 61 calls
`scatter_add`. MetaAtt additionally needs `scatter_softmax`, which the fallback
does not provide. This is the source of the 11 long-standing test failures.
Either fix the fallback imports or de-register the encoder — a broken option is
worse than an absent one.
