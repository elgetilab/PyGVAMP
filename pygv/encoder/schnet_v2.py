"""
SchNet encoder v2 — paper-faithful per-atom ReLU before global pooling.

The v1 encoder (pygv/encoder/schnet.py:SchNetEncoderNoEmbed) feeds the
residual-stacked node features straight into global_mean_pool.  The
Ghorbani 2022 reference (github.com/ghorbanimahdi73/GraphVampNet,
src/model.py:337) instead applies a per-atom ReLU between the residual
loop and the pooling step:

    for idx in range(self.n_conv):
        tmp_conv, _ = self.convs[idx](...)
        atom_emb = atom_emb + tmp_conv          # residual

    emb = self.conv_activation(atom_emb)         # nn.ReLU(), per-atom
    # pool / amino_emb / softmax

ReLU and mean-pool do not commute, so this is not equivalent to applying
an activation after pooling — it gates negative per-atom activations
before they are averaged.  See SCHNET_VERSIONS.md for context.
"""
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from pygv.encoder.schnet import SchNetEncoderNoEmbed


class SchNetEncoderNoEmbedV2(SchNetEncoderNoEmbed):
    """SchNet encoder with paper-faithful per-atom ReLU before pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_conv_activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch=None):
        import torch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        attentions = []
        for interaction in self.interactions:
            delta, attention = interaction(h, edge_index, edge_attr)
            h = h + delta
            if attention is not None:
                attentions.append(attention)

        h = self.post_conv_activation(h)

        pooled = global_mean_pool(h, batch)
        output = self.output_network(pooled)
        return output, (h, attentions)
