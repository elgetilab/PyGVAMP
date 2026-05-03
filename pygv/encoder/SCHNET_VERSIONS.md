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
