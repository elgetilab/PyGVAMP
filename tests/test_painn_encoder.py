"""
PaiNN encoder — equivariance, invariance and the pipeline contract.

Why these tests exist (claude/PAINN_SCOPE.md §6 risk 4): **equivariance bugs are
silent**. A broken vector channel does not raise — it just degrades accuracy, and
you find out after a 30-hour cluster arm. Every property below is therefore pinned
before the encoder is allowed near a production run.

The two that matter most:
  * ``test_scalar_readout_is_rotation_invariant`` — VAMP states must not depend on
    each MD frame's arbitrary lab frame.
  * ``test_vector_channel_is_rotation_equivariant`` — this is the *mechanism*. If
    it silently degrades to invariant, PaiNN reduces to an expensive SchNet and
    the whole experiment measures nothing.
"""

import math
import pytest
import torch

from pygv.encoder.painn import PaiNNEncoder


NODE_DIM, EDGE_DIM, HIDDEN, OUT = 8, 16, 16, 12


def _rotation(axis=(0.3, -0.7, 0.5), angle=0.9):
    k = torch.tensor(axis, dtype=torch.float64)
    k = k / k.norm()
    K = torch.tensor([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]],
                     dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R.float()


def _reflection():
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def _system(n=10, seed=0):
    """Random-ish 3D system with a k-NN graph, in the dataset's [target, source] convention."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * 3.0
    x = torch.randn(n, NODE_DIM, generator=g)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float('inf'))
    k = min(4, n - 1)
    _, nn = torch.topk(d, k, dim=1, largest=False)
    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    tgt = nn.reshape(-1)
    edge_index = torch.stack([tgt, src], dim=0)
    # Gaussian-expanded distances, mirroring what VAMPNetDataset produces
    dist = d.clamp(max=1e6)[src, tgt]
    centers = torch.linspace(0, 10, EDGE_DIM)
    edge_attr = torch.exp(-((dist.unsqueeze(1) - centers.unsqueeze(0)) ** 2) / 2.0)
    return x, pos, edge_index, edge_attr


def _encoder(seed=0, **kw):
    torch.manual_seed(seed)
    enc = PaiNNEncoder(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=HIDDEN,
                       output_dim=OUT, n_interactions=3, **kw)
    return enc.eval()


# ---------------------------------------------------------------------------
# 1. the properties the science depends on
# ---------------------------------------------------------------------------

def test_scalar_readout_is_rotation_invariant():
    """Rotating the whole system must not change the encoder output.

    MD frames are not aligned to any common frame, so a rotation-dependent
    readout would make the VAMP states depend on an arbitrary lab frame.
    """
    enc = _encoder()
    x, pos, ei, ea = _system()
    R = _rotation()

    with torch.no_grad():
        out_a, _ = enc(x, ei, ea, batch=None, pos=pos)
        out_b, _ = enc(x, ei, ea, batch=None, pos=pos @ R.T)

    assert torch.allclose(out_a, out_b, atol=1e-4), (
        f"readout changed under rotation (max diff {(out_a - out_b).abs().max():.2e})"
    )


def test_scalar_readout_is_translation_invariant():
    enc = _encoder()
    x, pos, ei, ea = _system()
    with torch.no_grad():
        out_a, _ = enc(x, ei, ea, batch=None, pos=pos)
        out_b, _ = enc(x, ei, ea, batch=None, pos=pos + torch.tensor([11.0, -3.0, 4.0]))
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_vector_channel_is_rotation_equivariant():
    """v(Rx) == R v(x) — the actual mechanism, and the thing that breaks silently.

    If the vector channel were merely invariant, PaiNN would be an expensive
    SchNet and the comparison would measure nothing.
    """
    enc = _encoder()
    x, pos, ei, ea = _system()
    R = _rotation()

    with torch.no_grad():
        _, aux_a = enc(x, ei, ea, batch=None, pos=pos, return_vectors=True)
        _, aux_b = enc(x, ei, ea, batch=None, pos=pos @ R.T, return_vectors=True)
    v_a, v_b = aux_a['v'], aux_b['v']          # (n, 3, F)

    rotated = torch.einsum('ij,njf->nif', R, v_a)
    assert torch.allclose(rotated, v_b, atol=1e-4), (
        f"vector channel is not equivariant (max diff {(rotated - v_b).abs().max():.2e})"
    )


def test_vector_channel_is_actually_used():
    """Guards against a vector channel that stays at its zero init.

    PaiNN initialises v = 0. If the message block failed to write into it, every
    property test above would still pass — trivially, on an all-zero tensor.
    """
    enc = _encoder()
    x, pos, ei, ea = _system()
    with torch.no_grad():
        _, aux = enc(x, ei, ea, batch=None, pos=pos, return_vectors=True)
    assert aux['v'].abs().max() > 1e-6, "vector channel never left its zero init"


def test_scalar_readout_is_reflection_invariant_documents_a_limitation():
    """PaiNN's scalar readout is reflection-EVEN — it does not see chirality.

    The vector channel couples to scalars only through orthogonally-invariant
    contractions (norms and dot products), and <Rv, Rw> = <v, w> holds for
    improper R too. So standard PaiNN with a scalar readout is blind to
    handedness, exactly like the distance map.

    This REFINES claude/PAINN_SCOPE.md §1: PaiNN's gain over a distance-only
    descriptor is *angular resolution*, NOT chirality. Recorded as a test so the
    scope's chirality framing is not accidentally revived in a write-up.
    """
    enc = _encoder()
    x, pos, ei, ea = _system()
    with torch.no_grad():
        out_a, _ = enc(x, ei, ea, batch=None, pos=pos)
        out_b, _ = enc(x, ei, ea, batch=None, pos=pos @ _reflection().T)
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_output_differs_from_a_distance_only_baseline():
    """Sanity: the encoder's output actually depends on directions, not just |r|.

    Two systems with identical distance matrices but different geometry must give
    different outputs, or the vector channel is contributing nothing.
    """
    enc = _encoder()
    x, pos, ei, ea = _system()
    # Permuting coordinates changes directions while leaving all distances intact
    pos2 = pos[:, [1, 2, 0]]
    with torch.no_grad():
        a, _ = enc(x, ei, ea, batch=None, pos=pos)
        b, _ = enc(x, ei, ea, batch=None, pos=pos2)
    # A coordinate permutation is orthogonal, so an invariant readout gives the
    # SAME answer; that is expected and fine. What must hold is that a genuinely
    # different geometry with the same edge_attr changes the output.
    pos3 = pos.clone()
    pos3[0] = -pos3[0]
    with torch.no_grad():
        c, _ = enc(x, ei, ea, batch=None, pos=pos3)
    assert not torch.allclose(a, c, atol=1e-6), (
        "moving an atom (same edge_attr) did not change the output — the "
        "positional pathway is dead"
    )


# ---------------------------------------------------------------------------
# 2. pipeline contract
# ---------------------------------------------------------------------------

def test_forward_returns_the_encoder_contract():
    """(output, (node_features, attentions)) — what VAMPNet.forward unpacks."""
    enc = _encoder()
    x, pos, ei, ea = _system()
    with torch.no_grad():
        out, aux = enc(x, ei, ea, batch=None, pos=pos)
    assert out.shape == (1, OUT)
    assert isinstance(aux, tuple) and len(aux) == 2
    h, attentions = aux
    assert h.shape == (x.shape[0], HIDDEN)
    assert isinstance(attentions, list), "PaiNN has no attention; must still be a list"


def test_declares_requires_pos():
    """VAMPNet passes pos only to encoders that ask for it."""
    assert getattr(_encoder(), 'requires_pos', False) is True


def test_batched_graphs_are_independent():
    """Two graphs in one batch must give the same answer as run separately."""
    enc = _encoder()
    x1, p1, e1, a1 = _system(n=8, seed=1)
    x2, p2, e2, a2 = _system(n=6, seed=2)

    x = torch.cat([x1, x2])
    pos = torch.cat([p1, p2])
    ei = torch.cat([e1, e2 + x1.shape[0]], dim=1)
    ea = torch.cat([a1, a2])
    batch = torch.cat([torch.zeros(x1.shape[0], dtype=torch.long),
                       torch.ones(x2.shape[0], dtype=torch.long)])

    with torch.no_grad():
        out_batched, _ = enc(x, ei, ea, batch=batch, pos=pos)
        o1, _ = enc(x1, e1, a1, batch=None, pos=p1)
        o2, _ = enc(x2, e2, a2, batch=None, pos=p2)

    assert out_batched.shape == (2, OUT)
    assert torch.allclose(out_batched[0], o1[0], atol=1e-5)
    assert torch.allclose(out_batched[1], o2[0], atol=1e-5)


def test_gradients_reach_every_parameter():
    """A parameter with no gradient is a wiring bug, not a design choice.

    Uses two graphs because the readout MLP inherits PyG's default batch_norm
    (same as SchNet's output_network), which rejects a batch of 1 in train mode.
    """
    enc = _encoder()
    enc.train()
    x1, p1, e1, a1 = _system(n=8, seed=1)
    x2, p2, e2, a2 = _system(n=6, seed=2)
    x = torch.cat([x1, x2])
    pos = torch.cat([p1, p2])
    ei = torch.cat([e1, e2 + x1.shape[0]], dim=1)
    ea = torch.cat([a1, a2])
    batch = torch.cat([torch.zeros(x1.shape[0], dtype=torch.long),
                       torch.ones(x2.shape[0], dtype=torch.long)])
    out, _ = enc(x, ei, ea, batch=batch, pos=pos)
    out.sum().backward()
    dead = [n for n, p in enc.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"parameters received no gradient: {dead}"


def test_missing_pos_fails_loudly():
    """Silently falling back to a distance-only path would invalidate the arm."""
    enc = _encoder()
    x, pos, ei, ea = _system()
    with pytest.raises(ValueError, match="requires node positions"):
        enc(x, ei, ea, batch=None, pos=None)


def test_no_nan_on_coincident_atoms():
    """Zero-length displacement vectors must not produce NaN via 1/|r|."""
    enc = _encoder()
    x, pos, ei, ea = _system()
    pos[1] = pos[0]  # exactly coincident
    with torch.no_grad():
        out, _ = enc(x, ei, ea, batch=None, pos=pos)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. end-to-end through VAMPNet (the wiring, not just the encoder)
# ---------------------------------------------------------------------------

def _pyg_batch(n=10, seed=0, n_graphs=2):
    from torch_geometric.data import Data, Batch
    graphs = []
    for g in range(n_graphs):
        x, pos, ei, ea = _system(n=n, seed=seed + g)
        graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, pos=pos, num_nodes=n))
    return Batch.from_data_list(graphs)


def _vampnet_with_painn(n_states=4):
    from pygv.vampnet.vampnet import VAMPNet
    from pygv.scores.vamp_score_v0 import VAMPScore
    from pygv.classifier.SoftmaxMLP import SoftmaxMLP
    torch.manual_seed(0)
    enc = PaiNNEncoder(node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=HIDDEN,
                       output_dim=OUT, n_interactions=2)
    clf = SoftmaxMLP(in_channels=OUT, hidden_channels=HIDDEN,
                     out_channels=n_states, num_layers=2)
    return VAMPNet(encoder=enc, vamp_score=VAMPScore(epsilon=1e-6, mode='regularize'),
                   classifier_module=clf)


def test_vampnet_forward_passes_pos_to_painn():
    """VAMPNet must hand `pos` to encoders declaring requires_pos.

    Without this the encoder raises, so a green test here is the wiring check.
    """
    model = _vampnet_with_painn().eval()
    with torch.no_grad():
        probs = model(_pyg_batch())
    if isinstance(probs, tuple):
        probs = probs[0]
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5), "softmax rows must sum to 1"


def test_vampnet_backward_through_painn():
    """Gradients must flow end to end, or training silently does nothing.

    NB the loss must not be ``probs.sum()``: softmax rows sum to 1, so that
    quantity is exactly n_graphs regardless of the parameters and its gradient is
    mathematically zero — a degenerate objective that looks like a wiring bug.
    """
    model = _vampnet_with_painn()
    model.train()
    out = model(_pyg_batch(n_graphs=4))
    if isinstance(out, tuple):
        out = out[0]
    (out ** 2).sum().backward()
    enc_grads = [p.grad is not None and p.grad.abs().sum() > 0
                 for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert all(enc_grads), "some PaiNN parameters got no gradient through VAMPNet"


def test_vampnet_raises_when_graph_has_no_pos():
    """A dataset built before `pos` existed must fail loudly, not silently.

    Silently falling back would make a PaiNN arm secretly a SchNet arm.
    """
    from torch_geometric.data import Data, Batch
    x, pos, ei, ea = _system()
    g = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=x.shape[0])  # no pos
    model = _vampnet_with_painn().eval()
    with pytest.raises(ValueError, match="requires node positions"):
        model(Batch.from_data_list([g, g]))


def test_distance_only_encoders_are_unaffected_by_the_pos_plumbing():
    """SchNet must not receive pos, and must behave exactly as before."""
    from pygv.encoder.schnet import SchNetEncoderNoEmbed
    from pygv.vampnet.vampnet import VAMPNet
    from pygv.scores.vamp_score_v0 import VAMPScore
    from pygv.classifier.SoftmaxMLP import SoftmaxMLP
    torch.manual_seed(0)
    enc = SchNetEncoderNoEmbed(node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                               hidden_dim=HIDDEN, output_dim=OUT, n_interactions=2)
    assert getattr(enc, 'requires_pos', False) is False
    model = VAMPNet(encoder=enc, vamp_score=VAMPScore(epsilon=1e-6, mode='regularize'),
                    classifier_module=SoftmaxMLP(in_channels=OUT, hidden_channels=HIDDEN,
                                                 out_channels=4, num_layers=2)).eval()
    with torch.no_grad():
        out = model(_pyg_batch())
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# 4. analysis must survive an encoder with no attention
# ---------------------------------------------------------------------------

def test_attention_maps_skip_gracefully_when_there_is_no_attention():
    """PaiNN has no attention, so `attentions`/`edge_indices` come back all-None.

    Before the fix this raised "Could not determine the number of atoms from edge
    indices", Phase 3 swallowed it, and the run exited 0 with
    `analysis_completed: []` — the exact exit-0 masking pattern from job 877.
    Every PaiNN run would have produced no analysis at all.

    Contract: return (None, populations) rather than raising, so the caller can
    skip attention artifacts and still emit ITS/CK/transition matrices/state
    structures.
    """
    import numpy as np
    from pygv.utils.analysis import calculate_state_edge_attention_maps

    n_frames, n_states = 20, 4
    probs = np.random.default_rng(0).random((n_frames, n_states))
    probs /= probs.sum(axis=1, keepdims=True)

    maps, pops = calculate_state_edge_attention_maps(
        edge_attentions=[None] * n_frames,
        edge_indices=[None] * n_frames,
        probs=probs,
        save_dir=None,
        protein_name='test',
    )

    assert maps is None, "no attention must yield None, not fabricated maps"
    assert pops is not None and len(pops) == n_states
    assert abs(float(np.sum(pops)) - 1.0) < 1e-6, "populations must still be computed"


# ---------------------------------------------------------------------------
# 5. PaiNN must survive the pipeline's weight re-initialisation
# ---------------------------------------------------------------------------

def test_painn_is_not_destroyed_by_init_for_vamp():
    """`init_for_vamp` must not re-initialise PaiNN into producing NaNs.

    Root cause of the alpha3D collapse (job 922, 2026-08-23). training.py:436
    applies init_for_vamp(model, 'kaiming_normal') to the WHOLE model. Its
    graph-model detection matches 'GCN|GAT|GraphConv|GIN|EdgeConv' in module type
    names — SchNet's GCNInteraction matches, PaiNN's modules match nothing, so the
    two encoders take different init paths. Kaiming re-init blew up PaiNN's
    residual scalar/vector accumulation across 4 blocks, the forward produced NaN,
    and VAMPNet's guard silently rewrote NaN -> 1e-6, giving a constant output and
    VAMP-2 = 1.0000 (the degenerate value) for all 100 epochs. The run still
    exited 0 with a complete analysis, and logged 37,762 NaN warnings that a
    grep for 'Error|Traceback' does not catch.

    PaiNN declares `self_initialized`; the training path must honour it.
    """
    import torch
    from pygv.utils.nn_utils import init_for_vamp
    import io, contextlib

    assert getattr(PaiNNEncoder, 'self_initialized', False) is True, (
        "PaiNNEncoder must declare self_initialized so the pipeline preserves its init"
    )

    torch.manual_seed(0)
    enc = _encoder()
    before = {k: v.clone() for k, v in enc.state_dict().items()}

    from pygv.pipe.training import apply_vamp_init
    with contextlib.redirect_stdout(io.StringIO()):
        apply_vamp_init(enc, method='kaiming_normal')

    after = enc.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), (
        "apply_vamp_init modified a self-initialised encoder's weights"
    )

    # and the forward is still finite
    x, pos, ei, ea = _system()
    with torch.no_grad():
        out, _ = enc(x, ei, ea, batch=None, pos=pos)
    assert torch.isfinite(out).all()


def test_apply_vamp_init_still_initialises_normal_encoders():
    """The guard must be narrow: SchNet and friends must still be re-initialised."""
    import torch, io, contextlib
    from pygv.encoder.schnet import SchNetEncoderNoEmbed
    from pygv.pipe.training import apply_vamp_init

    torch.manual_seed(0)
    enc = SchNetEncoderNoEmbed(node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                               hidden_dim=HIDDEN, output_dim=OUT, n_interactions=2)
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    with contextlib.redirect_stdout(io.StringIO()):
        apply_vamp_init(enc, method='kaiming_normal')
    after = enc.state_dict()
    changed = sum(0 if torch.equal(before[k], after[k]) else 1 for k in before)
    assert changed > 0, "apply_vamp_init became a no-op for a normal encoder"
