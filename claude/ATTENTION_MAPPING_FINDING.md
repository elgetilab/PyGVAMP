# Attention-flag mapping was wrong from v1 onward (2026-05-06)

## Summary

Every villin reproduction run (v1 through v9) was launched with
`--no_use_attention`, based on the mapping recorded in
`VILLIN_REPRO_LOG.md:125`:

> `--no_use_attention` | Original GraphVAMPNet uses classic SchNet, no attention

**That mapping was wrong.**  The Ghorbani 2022 reference's "SchNet"
branch (selected by `--conv_type SchNet` in `gpu_1.sh`) is **not**
vanilla SchNet — its `InteractionBlock` contains a softmax-attention
mechanism over neighbors with a learnable parameter.  Our v4–v9 runs
disabled the structurally-equivalent attention in our codebase.

## Evidence

### Reference: `GraphVampNet/src/layers.py` (`InteractionBlock` → `ContinuousFilterConv.forward`)

```python
nbr_filter = torch.matmul(conv_features, self.nbr_filter).view([n_batch, n_atoms, n_neighbors])
nbr_filter = F.softmax(nbr_filter, -1).to(device)
```

`self.nbr_filter` is a learnable attention parameter.  The `attn_probs`
returned from `InteractionBlock.forward()` (which the model exposes as
`tmp_conv, attn_probs = self.convs[idx](features=atom_emb, ...)`) **is**
the softmax-attention output across neighbors.  This is what the
paper's attention plots visualize.

### Ours: `pygv/encoder/schnet.py:64-68, 129-135`

```python
if use_attention:
    self.attention_vector = nn.Parameter(torch.Tensor(out_channels, 1))
    nn.init.xavier_uniform_(self.attention_vector, gain=1.414)

# In message():
if self.use_attention:
    attention = torch.matmul(messages, self.attention_vector).squeeze(-1)
    normalized_attention = softmax(attention, edge_index_i)
    # re-weight messages by attention
```

Same shape: single learnable vector, matmul, softmax over neighbors per
target node, element-wise re-weighting.  **Our `use_attention=True` is
the structural match for the reference's `InteractionBlock`, not
`use_attention=False`.**

### Why it took so long to surface

1. The reference's flag is named `--conv_type SchNet`, suggesting
   "vanilla SchNet, no attention" to anyone who doesn't open
   `layers.py`.  The attention is hidden inside `ContinuousFilterConv`.
2. Our flag is named `--use_attention`, suggesting it's an optional
   *extra* layer.  Both projects independently misled.
3. `tests/test_encoders.py:312-322` (`test_attention_disabled`) only
   verifies our encoder *runs* with attention off; no equivalence claim
   to the reference.
4. `tests/test_ml3_equivalence.py` is for ML3, not SchNet, and tests
   our-without-attention == reference-without-attention — orthogonal
   to this question.
5. **No SchNet-vs-reference numerical equivalence test exists.**  The
   user's recollection of "tests confirming SchNet is correct" is
   partial — internal correctness yes, paper equivalence no.

## What the paper text said (informally, recalled by user)

> "they mention attention values as an additional layer for the
> important residues visualization"

The user's intuition that the paper used attention was correct.  The
paper text does describe attention as part of the model.

## What this implies for the residual ~0.10 VAMP-2 gap

The gap to the paper's 3.78 (vs our v4 at 3.7126) is now consistent
with two independent contributions:

1. **Architectural deficit (newly identified)**: missing attention layer
   reduces the model's capacity to weight neighbors discriminatively.
   This could plausibly cost ~0.05-0.10 VAMP-2.
2. **τ-normalization (still true, but smaller than thought)**: VAMP-2 =
   1 + Σ σᵢ² is mechanically larger at shorter τ regardless of
   architecture.  v9 demonstrated 3.71 → 3.90 by changing τ alone.

The τ-finding from v9 is still a real observation.  But it may have
been over-interpreted as the *primary* explanation; (1) was sitting
underneath unnoticed.

## Followup: v10

Re-run v4 with `--use_attention` (single change).  Decision rule:

- v10 ≳ 3.76 → attention was the missing piece; gap closed
- v10 ~3.72–3.76 → partial improvement; combine with τ-norm to interpret
- v10 ≲ 3.72 → attention isn't the gap either; the τ-norm analysis is
  the only remaining story

If v10 closes the gap, the publication framing changes substantially:
the entire reproduction succeeds, and v6/v7/v8 (RBF + h_g probes) are
correctly classified as "ruled out" rather than "ruled out under the
wrong baseline".

## Side issues exposed by this audit

1. **Analysis pipeline bug** (independent of this finding): when no
   attention is computed, `edge_indices` stays empty and step 7 of the
   analysis raises, taking down steps 9–14 (including ITS, CK, HTML
   report) via the blanket try/except at `master_pipeline.py:226-232`.
   With `--use_attention`, this won't fire — `_attention_weights`
   gets set in the forward pass.  Still worth fixing for robustness,
   but no longer urgent for v10.

2. **Mismapping in VILLIN_REPRO_LOG.md:125**: should be corrected
   in-place with a note flagging the error.  The version logs are
   the historical record; rewriting them silently would lose the
   audit trail.

3. **No SchNet/InteractionBlock numerical equivalence test.**  This
   should exist for the same reason `test_ml3_equivalence.py` exists.
   Once v10 either closes the gap or doesn't, we'll know whether to
   prioritize this test.