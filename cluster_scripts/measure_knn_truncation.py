#!/usr/bin/env python
"""
How much information does the k-NN graph discard, and does it matter?

WHY (2026-08-23). Every encoder we have consumes Gaussian-expanded Cα distances
over a k-NN graph — NOT a full distance matrix. A full distance matrix fixes a
structure up to reflection; a k-NN-truncated one does not. PaiNN's directional
channel can only help if the truncation actually discards state-relevant geometry.

The angular pre-test came back null on Trp-cage (20 CA, k=7) — the system where the
distance map is LEAST truncated of anything we run. So that null may say more about
Trp-cage than about angular information in general. This script measures the
truncation directly, so a target system is chosen on evidence rather than intuition.

CALIBRATION IS THE POINT. Trp-cage/Villin/GTT/NTL9 are run as controls: we already
know the angular answer there is null. A metric that does not separate the
candidates from those controls is not measuring anything useful, and the honest
conclusion would be that PaiNN is unmotivated everywhere.

METRICS
  1. retention          fraction of atom pairs kept by the k-NN graph (per frame)
  2. hop coverage       fraction of pairs reachable within `n_interactions` hops.
                        THIS IS THE KEY CAVEAT: message passing propagates over
                        multiple hops, so a pair absent from the graph is not
                        necessarily invisible to the network. If hop coverage is
                        ~100%, the "truncation" argument is much weaker than raw
                        retention suggests.
  3. unexplained var    R^2 of a linear fit predicting DISCARDED pair distances
                        from RETAINED ones, over frames. 1 - R^2 is the fraction of
                        long-range variation the retained set cannot account for
                        even in principle. High = genuinely missing information.
  4. state leverage     cluster frames on RETAINED distances only, then measure how
                        much of the DISCARDED distance variance survives within
                        clusters. High = conformations the retained view calls
                        identical actually differ in long-range geometry, which is
                        exactly the gap an angular/equivariant encoder could fill.

Usage:
  python cluster_scripts/measure_knn_truncation.py --systems all
"""

import argparse
import json
import os
import sys

import numpy as np


def load_frames(traj_glob, top, n_frames, stride, selection='name CA'):
    import mdtraj as md
    from glob import glob
    files = sorted(glob(traj_glob))
    if not files:
        raise FileNotFoundError(f"no trajectories matched {traj_glob}")
    top_obj = md.load_topology(top)
    idx = top_obj.select(selection)
    out = []
    for f in files:
        for chunk in md.iterload(f, top=top, chunk=2000, stride=stride, atom_indices=idx):
            out.append(chunk.xyz)
            if sum(c.shape[0] for c in out) >= n_frames:
                break
        if sum(c.shape[0] for c in out) >= n_frames:
            break
    xyz = np.concatenate(out, axis=0)[:n_frames]
    return xyz.astype(np.float64)


def knn_mask(D, k):
    """Boolean (n,n) mask of retained directed edges, mirrored to undirected pairs."""
    n = D.shape[0]
    Di = D.copy()
    np.fill_diagonal(Di, np.inf)
    kk = min(k, n - 1)
    nn = np.argpartition(Di, kk - 1, axis=1)[:, :kk]
    M = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), kk)
    M[rows, nn.reshape(-1)] = True
    return M | M.T          # undirected: a pair survives if either endpoint keeps it


def hop_coverage(M, hops):
    """Fraction of pairs reachable within `hops` steps — the receptive field."""
    n = M.shape[0]
    reach = M.copy()
    frontier = M.copy()
    for _ in range(hops - 1):
        frontier = (frontier @ M) > 0
        reach |= frontier
    iu = np.triu_indices(n, 1)
    return float(reach[iu].mean())


def analyse(xyz, k, hops, n_clusters=6, seed=0):
    n_frames, n_atoms, _ = xyz.shape
    iu = np.triu_indices(n_atoms, 1)

    # Pairwise distances for every frame
    diff = xyz[:, :, None, :] - xyz[:, None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))                    # (frames, n, n)
    Dv = D[:, iu[0], iu[1]]                             # (frames, n_pairs)

    # Per-frame retention + a consensus mask (pairs retained in >50% of frames)
    retentions, cov = [], []
    keep_count = np.zeros(len(iu[0]))
    for f in range(n_frames):
        M = knn_mask(D[f], k)
        mv = M[iu[0], iu[1]]
        retentions.append(mv.mean())
        keep_count += mv
        if f < 20:                                       # hop coverage is expensive
            cov.append(hop_coverage(M, hops))
    retention = float(np.mean(retentions))
    coverage = float(np.mean(cov))

    consensus = keep_count > (0.5 * n_frames)
    n_ret, n_dis = int(consensus.sum()), int((~consensus).sum())
    if n_ret < 2 or n_dis < 2:
        return dict(retention=retention, hop_coverage=coverage,
                    note="degenerate split; too few retained or discarded pairs")

    R = Dv[:, consensus]                                 # retained view
    Q = Dv[:, ~consensus]                                # discarded view

    # --- 3. how much of Q is linearly predictable from R? -------------------
    # MUST be evaluated on HELD-OUT frames. Retained pairs outnumber frames on the
    # larger systems (a3d: 423 predictors vs 400 samples), so an in-sample fit is
    # perfect by construction and reports ~0 unexplained variance for every big
    # protein — an artifact that inverts the conclusion. Caught 2026-08-23.
    n_tr = int(0.7 * n_frames)
    if n_tr < 20 or (n_frames - n_tr) < 20:
        return dict(retention=retention, hop_coverage=coverage,
                    note="too few frames for a held-out fit")
    mu_R, mu_Q = R[:n_tr].mean(0), Q[:n_tr].mean(0)
    Rtr, Qtr = R[:n_tr] - mu_R, Q[:n_tr] - mu_Q
    Rte, Qte = R[n_tr:] - mu_R, Q[n_tr:] - mu_Q
    # Ridge, strength chosen by simple hold-out over a small grid.
    best = None
    for lam in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        A = Rtr.T @ Rtr + lam * np.eye(Rtr.shape[1])
        W = np.linalg.solve(A, Rtr.T @ Qtr)
        ss_res = float(((Qte - Rte @ W) ** 2).sum())
        if best is None or ss_res < best[0]:
            best = (ss_res, lam)
    ss_res, lam_best = best
    ss_tot = float((Qte ** 2).sum())
    unexplained = ss_res / ss_tot if ss_tot > 0 else 0.0
    overdet = Rtr.shape[1] >= n_tr   # predictors >= training samples

    # --- 4. state leverage: does Q vary WITHIN clusters defined by R? -------
    from sklearn.cluster import KMeans
    Rall = R - R.mean(0)
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(
        (Rall / (Rall.std(0) + 1e-9)))
    lab = km.labels_
    within = 0.0
    for c in np.unique(lab):
        sel = lab == c
        if sel.sum() < 2:
            continue
        within += ((Q[sel] - Q[sel].mean(0)) ** 2).sum()
    tot_all = float(((Q - Q.mean(0)) ** 2).sum())
    leverage = within / tot_all if tot_all > 0 else 0.0

    return dict(
        n_atoms=n_atoms, n_frames=n_frames, k=k,
        retention=retention,
        hop_coverage=coverage,
        unexplained_var=float(unexplained),
        ridge_lambda=float(lam_best),
        predictors_ge_samples=bool(overdet),
        state_leverage=float(leverage),
        n_retained_pairs=n_ret, n_discarded_pairs=n_dis,
    )


SYSTEMS = {
    # controls — angular pre-test was NULL on trpcage; these calibrate the metric
    'trpcage': dict(glob='/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/*.dcd',
                    top='/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb', k=7),
    'villin':  dict(glob='/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/*.dcd',
                    top='/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb', k=10),
    'gtt':     dict(glob='/mnt/hdd/data/gtt/DESRES-Trajectory_GTT-0-c-alpha/GTT-0-c-alpha/*.dcd',
                    top='/mnt/hdd/data/gtt/topol.pdb', k=10),
    'ntl9':    dict(glob='/mnt/hdd/data/ntl9/DESRES-Trajectory_NTL9-0-c-alpha/NTL9-0-c-alpha/*.dcd',
                    top='/mnt/hdd/data/ntl9/topol.pdb', k=10),
    # candidates
    'a3d':     dict(glob='/mnt/hdd/data/painn_candidates/DESRES-Trajectory_A3D-0-c-alpha/A3D-0-c-alpha/*.dcd',
                    top=None, k=10),
    'nug2':    dict(glob='/mnt/hdd/data/painn_candidates/DESRES-Trajectory_NuG2-0-c-alpha/NuG2-0-c-alpha/*.dcd',
                    top=None, k=10),
    'lambda':  dict(glob='/mnt/hdd/data/painn_candidates/DESRES-Trajectory_lambda-0-c-alpha/lambda-0-c-alpha/*.dcd',
                    top=None, k=10),
}


def find_top(cfg, name):
    if cfg['top'] and os.path.isfile(cfg['top']):
        return cfg['top']
    from glob import glob as _g
    base = os.path.dirname(os.path.dirname(cfg['glob']))
    for pat in ('*.pdb', '**/*.pdb'):
        hits = _g(os.path.join(base, pat), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no topology found for {name} under {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='all')
    ap.add_argument('--n-frames', type=int, default=1500)
    ap.add_argument('--stride', type=int, default=200)
    ap.add_argument('--hops', type=int, default=4, help='n_interactions in the encoders')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    names = list(SYSTEMS) if args.systems == 'all' else args.systems.split(',')
    rows = {}
    for name in names:
        cfg = SYSTEMS[name]
        try:
            top = find_top(cfg, name)
            xyz = load_frames(cfg['glob'], top, args.n_frames, args.stride)
            rows[name] = analyse(xyz, cfg['k'], args.hops)
            print(f"[ok] {name}: {rows[name]}", flush=True)
        except Exception as e:
            print(f"[skip] {name}: {e}", flush=True)

    print("\n" + "=" * 96)
    print(f"{'system':10} {'atoms':>6} {'k':>3} {'retention':>10} {'hop_cov':>8} "
          f"{'unexpl_var':>11} {'state_lev':>10} {'overdet':>8}")
    print("-" * 96)
    for n, r in rows.items():
        if 'n_atoms' not in r:
            continue
        print(f"{n:10} {r['n_atoms']:6} {r['k']:3} {r['retention']:10.3f} "
              f"{r['hop_coverage']:8.3f} {r['unexplained_var']:11.4f} {r['state_leverage']:10.4f} "
              f"{str(r['predictors_ge_samples']):>8}")
    print("=" * 96)
    print("controls (trpcage/villin/gtt/ntl9) are systems where the angular pre-test")
    print("was null or the landscape is known two-state. A candidate only justifies a")
    print("PaiNN arm if it separates CLEARLY from those on unexpl_var / state_lev.")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == '__main__':
    main()
