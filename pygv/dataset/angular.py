"""
Angular node features for Cα graphs.

Why this exists (claude/PAINN_SCOPE.md §7): GIN and ML3 changed the *aggregation*
over the same two-body distance inputs and neither beat SchNet on any system. The
untested axis is the *information* in the descriptor. These features add explicit
angular structure to the existing SchNet path, as a cheap falsification test before
committing ~30 runs to PaiNN.

Two independent groups, selectable via ``--angular_features``:

``chain``
    Cα pseudo bond angle θ and **signed** pseudo dihedral τ — the canonical
    internal coordinates of a Cα trace. The sign of τ carries handedness, which a
    pairwise distance matrix provably cannot represent (a structure and its mirror
    image have identical distance matrices). O(n) per frame.

``knn``
    A Gaussian-expanded distribution of cos(angle j–i–l) over pairs of i's k-NN
    neighbours — angular resolution over the same neighbourhood PaiNN's vector
    channel would see. NOTE this group is reflection-*even*, so it adds angular
    information but not chirality. O(n·k²) per frame.

All features are invariant under translation and proper rotation, so the VAMP
states remain independent of each MD frame's arbitrary lab frame.

Graphs are built on the fly in ``VAMPNetDataset.__getitem__``, so these run per
sample per epoch — keep them cheap and allocation-light.
"""

import torch


# Guards against division by ~0 for degenerate (collinear / coincident) geometry.
# Chosen well below any physical Cα separation (~3.8 Å) but far above float32 noise.
_EPS = 1e-8


def chain_angular_features(coords: torch.Tensor) -> torch.Tensor:
    """Pseudo bond angle and signed pseudo dihedral along the Cα chain.

    Parameters
    ----------
    coords : torch.Tensor
        ``(n_atoms, 3)`` coordinates in chain order.

    Returns
    -------
    torch.Tensor
        ``(n_atoms, 4)`` = ``[cos θ, sin θ, cos τ, sin τ]``.

        θ_i is the angle at atom i subtended by atoms i-1 and i+1, defined for
        ``1 <= i <= n-2``.  τ_i is the dihedral over atoms (i-1, i, i+1, i+2),
        defined for ``1 <= i <= n-3``.  Undefined positions are zero-padded, so
        the block is always ``(n_atoms, 4)`` regardless of chain length.
    """
    n = coords.shape[0]
    out = coords.new_zeros((n, 4))
    if n < 3:
        return out

    # Bond vectors b_i = r_{i+1} - r_i, for i = 0 .. n-2
    b = coords[1:] - coords[:-1]

    # --- pseudo bond angle at atom i (needs b_{i-1} and b_i) ---------------
    u = -b[:-1]                                    # i-1 -> i reversed: (i-1) - i
    v = b[1:]                                      # i -> i+1
    un = u / (u.norm(dim=1, keepdim=True) + _EPS)
    vn = v / (v.norm(dim=1, keepdim=True) + _EPS)
    cos_t = (un * vn).sum(dim=1).clamp(-1.0, 1.0)
    # sin θ >= 0 for θ in [0, π]; carries no sign information but keeps the
    # encoding smooth across θ = 0 / π.
    sin_t = torch.sqrt((1.0 - cos_t * cos_t).clamp_min(0.0))
    out[1:n - 1, 0] = cos_t
    out[1:n - 1, 1] = sin_t

    if n < 4:
        return out

    # --- signed pseudo dihedral over (i-1, i, i+1, i+2) --------------------
    # Standard normal-vector construction; the sign convention follows the usual
    # IUPAC atan2(( n1 x n2 )·b2_hat, n1·n2 ).
    b0, b1, b2 = b[:-2], b[1:-1], b[2:]
    n1 = torch.cross(b0, b1, dim=1)
    n2 = torch.cross(b1, b2, dim=1)
    b1n = b1 / (b1.norm(dim=1, keepdim=True) + _EPS)
    m1 = torch.cross(n1, b1n, dim=1)

    x = (n1 * n2).sum(dim=1)
    y = (m1 * n2).sum(dim=1)
    norm = torch.sqrt(x * x + y * y)
    # Collinear atoms make both normals vanish -> norm ~ 0; emit (0,0) there
    # rather than NaN, which is what the zero-init already encodes as "undefined".
    valid = norm > _EPS
    cos_d = torch.zeros_like(x)
    sin_d = torch.zeros_like(y)
    cos_d[valid] = x[valid] / norm[valid]
    sin_d[valid] = y[valid] / norm[valid]

    out[1:n - 2, 2] = cos_d
    out[1:n - 2, 3] = sin_d
    return out


def knn_angular_features(coords: torch.Tensor,
                         edge_index: torch.Tensor,
                         n_bins: int = 8) -> torch.Tensor:
    """Gaussian-expanded distribution of neighbour–centre–neighbour angles.

    For each atom i, takes every unordered pair (j, l) of i's graph neighbours and
    accumulates cos(angle j–i–l) into ``n_bins`` soft bins evenly spaced on
    ``[-1, 1]``.  Soft (Gaussian) rather than hard bins so the feature varies
    smoothly with geometry.

    Parameters
    ----------
    coords : torch.Tensor
        ``(n_atoms, 3)`` coordinates.
    edge_index : torch.Tensor
        ``(2, n_edges)`` in the dataset's ``[target, source]`` convention: column
        ``e`` means source ``edge_index[1, e]`` has neighbour ``edge_index[0, e]``.
    n_bins : int
        Number of soft bins.

    Returns
    -------
    torch.Tensor
        ``(n_atoms, n_bins)``.

    Notes
    -----
    Reflection-**even**: cos of an angle is unchanged by mirroring, so this group
    adds angular resolution but not handedness.  Only the signed dihedral in
    :func:`chain_angular_features` distinguishes a structure from its mirror image.
    """
    n = coords.shape[0]
    out = coords.new_zeros((n, n_bins))
    if edge_index.numel() == 0:
        return out

    tgt, src = edge_index[0], edge_index[1]
    centers = torch.unique(src)
    bin_centers = torch.linspace(-1.0, 1.0, n_bins, device=coords.device, dtype=coords.dtype)
    width = (2.0 / max(n_bins - 1, 1))
    gamma = 1.0 / (2.0 * width * width)

    for i in centers.tolist():
        nbrs = tgt[src == i]
        if nbrs.numel() < 2:
            continue
        d = coords[nbrs] - coords[i]
        d = d / (d.norm(dim=1, keepdim=True) + _EPS)
        cosm = (d @ d.T).clamp(-1.0, 1.0)
        # unordered pairs, excluding self-pairs
        iu = torch.triu_indices(cosm.shape[0], cosm.shape[0], offset=1, device=coords.device)
        cos_vals = cosm[iu[0], iu[1]]
        # soft-bin: (n_pairs, n_bins) -> sum over pairs
        expanded = torch.exp(-gamma * (cos_vals.unsqueeze(1) - bin_centers.unsqueeze(0)) ** 2)
        out[i] = expanded.sum(dim=0)

    return out


def angular_feature_dim(mode: str, n_bins: int = 8) -> int:
    """Number of columns :func:`build_angular_features` appends for ``mode``."""
    if mode in (None, 'none'):
        return 0
    if mode == 'chain':
        return 4
    if mode == 'knn':
        return n_bins
    if mode == 'both':
        return 4 + n_bins
    raise ValueError(f"Unknown angular_features mode: {mode!r}. "
                     "Choose from 'none', 'chain', 'knn', 'both'.")


def build_angular_features(mode: str,
                           coords: torch.Tensor,
                           edge_index: torch.Tensor,
                           n_bins: int = 8) -> torch.Tensor:
    """Assemble the angular block for ``mode``; ``(n_atoms, angular_feature_dim())``."""
    if mode in (None, 'none'):
        return coords.new_zeros((coords.shape[0], 0))
    if mode == 'chain':
        return chain_angular_features(coords)
    if mode == 'knn':
        return knn_angular_features(coords, edge_index, n_bins)
    if mode == 'both':
        return torch.cat([chain_angular_features(coords),
                          knn_angular_features(coords, edge_index, n_bins)], dim=1)
    raise ValueError(f"Unknown angular_features mode: {mode!r}. "
                     "Choose from 'none', 'chain', 'knn', 'both'.")
