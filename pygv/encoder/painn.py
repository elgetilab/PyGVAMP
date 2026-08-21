"""
PaiNN — polarizable/equivariant message passing encoder.

Reference: Schütt, Unke & Gastegger, "Equivariant message passing for the
prediction of tensorial properties and molecular spectra", ICML 2021.
Structure follows the schnetpack implementation (PaiNNInteraction + PaiNNMixing).

WHY THIS ENCODER EXISTS
-----------------------
SchNet, GIN and ML3 all consume the same two-body descriptor: Gaussian-expanded
Cα pairwise distances. They differ only in how they *aggregate* it, and none beat
SchNet on any system tested. PaiNN is the first encoder here that changes the
*inputs*: it maintains an equivariant vector channel v built from the unit
displacement vectors r̂_ij, so it carries directional structure a distance-only
representation does not contain.

WHAT IT DOES AND DOES NOT BUY (measured, see tests/test_painn_encoder.py)
-------------------------------------------------------------------------
It DOES add angular resolution: messages depend on the direction of r_ij, not
just |r_ij|, and directions accumulate across interaction blocks.

It does NOT add chirality. The vector channel couples back to scalars only
through orthogonally-invariant contractions — norms ‖v‖ and dot products ⟨v, w⟩ —
and ⟨Rv, Rw⟩ = ⟨v, w⟩ holds for improper R (det = −1) too. So the scalar readout
is reflection-EVEN and, like the distance map, blind to handedness. Detecting
handedness would need an odd-order invariant such as a triple product.
This is pinned by ``test_scalar_readout_is_reflection_invariant_documents_a_limitation``
and refines the framing in claude/PAINN_SCOPE.md §1.

SHAPES
------
    s : (n_atoms, F)        scalar (invariant) channel
    v : (n_atoms, 3, F)     vector (equivariant) channel, zero-initialised
Readout pools s only, which is what makes the graph embedding invariant.

INTEGRATION
-----------
``requires_pos = True`` tells VAMPNet to pass node positions through; every other
encoder keeps the original ``(x, edge_index, edge_attr, batch)`` signature.
PaiNN has no attention mechanism, so the returned attention list is always empty
— downstream analysis must skip attention artifacts rather than fail.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import MLP

# Guards 1/|r| for coincident atoms and ‖v‖ for a zero vector channel.  The
# vector channel starts at exactly zero, so this is hit on the first forward pass
# of every run, not just in pathological geometry.
_EPS = 1e-8


def _scatter_add(src, index, dim_size):
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``.

    Written against plain torch (``index_add_``) rather than torch_scatter, which
    is not installed in the deployed environment — the missing-dependency trap
    that left the Meta/MetaAtt encoders broken at runtime.
    """
    shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(shape)
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    return out.index_add_(0, index, src)


class PaiNNInteraction(nn.Module):
    """Equivariant message block.

    Scalar messages are the usual continuous-filter convolution.  Vector messages
    have two paths: one scales the *neighbour's* vector channel (``dv_v``), the
    other injects the edge direction r̂_ij (``dv_r``).  The second is what makes
    the block equivariant rather than merely invariant.
    """

    def __init__(self, n_features: int, edge_dim: int, activation: str = 'silu'):
        super().__init__()
        self.n_features = n_features
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU}.get(activation, nn.SiLU)

        # φ: per-atom context, split three ways (Δs, Δv_r, Δv_v)
        self.context_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            act(),
            nn.Linear(n_features, 3 * n_features),
        )
        # W: radial filter over the existing Gaussian-expanded edge features
        self.filter_net = nn.Linear(edge_dim, 3 * n_features)

    def forward(self, s, v, edge_index, edge_attr, dir_ij, cutoff=None):
        tgt, src = edge_index[0], edge_index[1]

        filters = self.filter_net(edge_attr)
        if cutoff is not None:
            filters = filters * cutoff.unsqueeze(-1)

        x = self.context_net(s)[tgt] * filters               # (e, 3F)
        ds, dv_r, dv_v = torch.split(x, self.n_features, dim=-1)

        # Vector message: scale the neighbour's vectors, plus a directional term.
        dv = dv_v.unsqueeze(1) * v[tgt] + dv_r.unsqueeze(1) * dir_ij.unsqueeze(-1)

        n = s.shape[0]
        return _scatter_add(ds, src, n), _scatter_add(dv, src, n)


class PaiNNMixing(nn.Module):
    """Intra-atomic update block: the only place vectors talk back to scalars.

    Both couplings — the norm ‖Vv‖ fed into the gate MLP, and the dot product
    ⟨Uv, Vv⟩ added to the scalar channel — are orthogonally invariant, which is
    what keeps the readout rotation-invariant while v stays equivariant.
    """

    def __init__(self, n_features: int, activation: str = 'silu'):
        super().__init__()
        self.n_features = n_features
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU}.get(activation, nn.SiLU)

        # Channel mixing over the feature axis only — no bias, so equivariance
        # survives (a bias would add a fixed vector that does not rotate).
        self.channel_mix = nn.Linear(n_features, 2 * n_features, bias=False)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            act(),
            nn.Linear(n_features, 3 * n_features),
        )

    def forward(self, s, v):
        # (n, 3, F) -> mix over F -> split into U and V
        mixed = self.channel_mix(v)
        u, w = torch.split(mixed, self.n_features, dim=-1)

        w_norm = torch.sqrt(torch.sum(w ** 2, dim=1) + _EPS)      # (n, F) invariant
        gate = self.gate_net(torch.cat([s, w_norm], dim=-1))      # (n, 3F)
        a_vv, a_sv, a_ss = torch.split(gate, self.n_features, dim=-1)

        dv = a_vv.unsqueeze(1) * u                                # equivariant
        ds = a_ss + a_sv * torch.sum(u * w, dim=1)                # invariant
        return ds, dv


class PaiNNEncoder(nn.Module):
    """PaiNN encoder with the project's standard encoder interface.

    Parameters
    ----------
    node_dim : int
        Input node feature width (auto-inferred from the dataset by the pipeline).
    edge_dim : int
        Width of the Gaussian-expanded edge features.
    hidden_dim : int
        Width of the scalar and vector channels (``F`` above).
    output_dim : int
        Width of the pooled graph embedding handed to the classifier.
    n_interactions : int
        Number of (message, update) block pairs.
    cutoff : float, optional
        Cosine cutoff radius in the same length units as the coordinates.  Default
        ``None`` = no cutoff, which is the right default here: the graphs are k-NN,
        not radius graphs, so a hard cutoff would silently delete legitimate
        neighbours that the k-NN construction deliberately kept.
    shared_interactions : bool
        Reuse one block for every interaction (parameter-efficient variant).
    """

    requires_pos = True   # VAMPNet passes node positions to encoders that ask

    def __init__(self, node_dim, edge_dim, hidden_dim=64, output_dim=32,
                 n_interactions=3, activation='silu', cutoff=None,
                 shared_interactions=False, use_attention=False):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.shared_interactions = shared_interactions
        if use_attention:
            # Accepted for CLI symmetry with the other encoders, but PaiNN has no
            # attention mechanism.  Warn rather than silently ignoring it.
            import warnings
            warnings.warn("PaiNNEncoder has no attention mechanism; "
                          "--use_attention is ignored for this encoder.")
        self.use_attention = False

        self.input_proj = nn.Linear(node_dim, hidden_dim)

        n_blocks = 1 if shared_interactions else n_interactions
        self.interactions = nn.ModuleList(
            [PaiNNInteraction(hidden_dim, edge_dim, activation) for _ in range(n_blocks)])
        self.mixings = nn.ModuleList(
            [PaiNNMixing(hidden_dim, activation) for _ in range(n_blocks)])

        self.output_network = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=2,
            act=activation if activation in ('relu', 'tanh') else 'relu',
        )

    def forward(self, x, edge_index, edge_attr, batch=None, pos=None,
                return_vectors=False):
        """
        Returns
        -------
        (output, aux)
            ``output`` is ``(n_graphs, output_dim)``.  ``aux`` is
            ``(node_features, attentions)`` to match the project's encoder
            contract — ``attentions`` is always ``[]`` because PaiNN has none.
            With ``return_vectors=True``, ``aux`` is a dict exposing the raw
            scalar/vector channels for the equivariance tests.
        """
        if pos is None:
            raise ValueError(
                "PaiNNEncoder requires node positions (`pos`), but received None. "
                "The dataset must attach `pos` to each graph. Falling back to a "
                "distance-only path would silently reduce PaiNN to SchNet."
            )
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        tgt, src = edge_index[0], edge_index[1]
        r_ij = pos[tgt] - pos[src]
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        dir_ij = r_ij / (d_ij + _EPS)          # unit displacement, equivariant

        cutoff = None
        if self.cutoff is not None:
            d = d_ij.squeeze(-1)
            cutoff = torch.where(
                d < self.cutoff,
                0.5 * (torch.cos(d * torch.pi / self.cutoff) + 1.0),
                torch.zeros_like(d),
            )

        s = self.input_proj(x)                                   # (n, F)
        v = torch.zeros(s.shape[0], 3, self.hidden_dim,
                        device=s.device, dtype=s.dtype)          # (n, 3, F)

        for i in range(self.n_interactions):
            block = 0 if self.shared_interactions else i
            ds, dv = self.interactions[block](s, v, edge_index, edge_attr, dir_ij, cutoff)
            s = s + ds
            v = v + dv
            ds, dv = self.mixings[block](s, v)
            s = s + ds
            v = v + dv

        pooled = global_mean_pool(s, batch)                      # invariant readout
        output = self.output_network(pooled)

        if return_vectors:
            return output, {'s': s, 'v': v}
        # PaiNN has no attention — empty list, so downstream code can detect it.
        return output, (s, [])
