"""
Multi-τ VAMP-2 transfer-evaluation scan for villin v4 and v10.

For each saved model (v4 = no attention, v10 = with attention, both at
SEED=0, both trained at τ=20 ns), runs inference on the FULL trajectory
to obtain a continuous chi(t) series, then computes VAMP-2 at a sweep
of lag times τ.  Identifies the τ at which each model's transfer-eval
VAMP-2 crosses the paper's 3.78.

WHY full inference (not transformed_traj.npz):
  The analysis pipeline saves a *random sub-sampled* chi (50000 of
  ~700k frames, with sorted-but-non-contiguous indices —
  pygv/pipe/analysis.py:133-138).  Slicing chi[:-k]/chi[k:] on that
  sub-sample produces pairs with variable physical Δt, not a clean τ.
  This script avoids that by computing chi for every frame in time
  order.

This is a *transfer evaluation* — both models were trained at τ=20 ns
and are being scored at others.  It bounds, but does not pinpoint,
the τ the paper actually trained at.

Outputs:
  /mnt/hdd/experiments/villin_tau_scan/chi_full_v4.npy
  /mnt/hdd/experiments/villin_tau_scan/chi_full_v10.npy
  /mnt/hdd/experiments/villin_tau_scan/villin_tau_scan.csv
  /mnt/hdd/experiments/villin_tau_scan/villin_tau_scan.png
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from pygv.dataset.vampnet_dataset import VAMPNetDataset
from pygv.vampnet.vampnet import VAMPNet
from pygv.scores.vamp_score_v0 import VAMPScore


# ---- Config ---------------------------------------------------------------

TIMESTEP_NS = 0.2  # DESRES physical step (200 ps/frame)
PAPER_VAMP2 = 3.78
V4_TRAIN_BEST = 3.7126   # at τ=20 ns
V10_TRAIN_BEST = 3.7298  # at τ=20 ns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("/mnt/hdd/experiments/villin_tau_scan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_DIR = "/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/"
TOP_FILE = "/mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb"
FILE_PATTERN = "2F4K-0-c-alpha-*.dcd"

RUNS = {
    "v4 (no attention)": {
        "exp_dir": "/mnt/hdd/experiments/villin_repro_v4/seed_00/exp_villin_20260429_215336",
        "color": "tab:orange",
    },
    "v10 (with attention)": {
        "exp_dir": "/mnt/hdd/experiments/villin_repro_v10/seed_00/exp_villin_20260506_235316",
        "color": "tab:blue",
    },
}

# τ grid in nanoseconds:
#   - 7 pts below 2 ns for context
#   - dense 0.5 ns spacing in the user's focus zone (2–20 ns)
#   - sparse log-ish above 20 ns
tau_low  = [0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8]
tau_mid  = [round(2.0 + 0.5 * i, 1) for i in range(37)]   # 2.0, 2.5, ..., 20.0
tau_high = [22, 25, 30, 35, 40, 50, 60, 80, 100, 140, 200]
TAU_NS = sorted(set(tau_low + tau_mid + tau_high))


# ---- Inference: get continuous chi(t) from full trajectory ---------------

def compute_chi_full(exp_dir: str, label: str) -> np.ndarray:
    """Forward every frame through the saved model in time order."""
    cache_dir = os.path.join(exp_dir, "cache")
    model_path = sorted(glob.glob(
        os.path.join(exp_dir, "training", "lag20.0ns_4states", "*", "models",
                     "best_model.pt")))[-1]
    print(f"\n[{label}] cache: {cache_dir}")
    print(f"[{label}] model: {model_path}")

    traj_files = sorted(glob.glob(os.path.join(TRAJ_DIR, FILE_PATTERN)))
    dataset = VAMPNetDataset(
        trajectory_files=traj_files,
        topology_file=TOP_FILE,
        lag_time=20.0,
        n_neighbors=10,
        selection="name CA",
        stride=1,
        cache_dir=cache_dir,
        use_cache=True,
        timestep=0.2,
        continuous=True,
    )
    print(f"[{label}] dataset frames: {dataset.n_frames}")

    frames_dataset = dataset.get_frames_dataset(return_pairs=False)
    loader = DataLoader(frames_dataset, batch_size=512, shuffle=False)

    model = VAMPNet.load_complete_model(model_path, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)

    chi_chunks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            chi, _ = model.forward(batch, apply_classifier=True)
            chi_chunks.append(chi.cpu())
            if i % 200 == 0:
                print(f"[{label}]   batch {i+1}/{len(loader)}")

    chi = torch.cat(chi_chunks).numpy().astype(np.float64)
    print(f"[{label}] chi_full shape: {chi.shape} "
          f"(populations: {chi.mean(0).round(4).tolist()})")
    return chi


# ---- Score evaluator ------------------------------------------------------

scorer = VAMPScore(method="VAMP2", epsilon=1e-6, mode="trunc")


def vamp2_at_tau(chi: torch.Tensor, k_frames: int) -> float:
    chi_t0 = chi[:-k_frames]
    chi_t1 = chi[k_frames:]
    with torch.no_grad():
        return scorer(chi_t0, chi_t1).item()


def crossover_tau(pts, target):
    for i in range(len(pts) - 1):
        t0, v0 = pts[i]
        t1, v1 = pts[i + 1]
        if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
            frac = (target - v0) / (v1 - v0)
            return t0 + frac * (t1 - t0)
    return None


# ---- Run --------------------------------------------------------------

results = {}  # label -> list of (tau_ns, vamp2)

for label, info in RUNS.items():
    chi_path = OUT_DIR / f"chi_full_{label.split()[0]}.npy"
    if chi_path.exists():
        print(f"\n[{label}] reusing cached {chi_path}")
        chi_np = np.load(chi_path)
    else:
        chi_np = compute_chi_full(info["exp_dir"], label)
        np.save(chi_path, chi_np)
        print(f"[{label}] saved {chi_path}")

    chi = torch.from_numpy(chi_np).to(torch.float64)

    pts = []
    for tau_ns in TAU_NS:
        k = max(1, int(round(tau_ns / TIMESTEP_NS)))
        v = vamp2_at_tau(chi, k)
        pts.append((tau_ns, v))
    results[label] = pts


# ---- CSV --------------------------------------------------------------

csv_path = OUT_DIR / "villin_tau_scan.csv"
with open(csv_path, "w") as f:
    f.write("tau_ns")
    for label in RUNS:
        f.write(f",{label}")
    f.write("\n")
    for i, tau_ns in enumerate(TAU_NS):
        f.write(f"{tau_ns}")
        for label in RUNS:
            f.write(f",{results[label][i][1]:.6f}")
        f.write("\n")
print(f"\nSaved CSV: {csv_path}")


# ---- Crossover ---------------------------------------------------------

print("\n=== Crossover (where VAMP-2 hits 3.78) ===")
crossovers = {}
for label in RUNS:
    tx = crossover_tau(results[label], PAPER_VAMP2)
    crossovers[label] = tx
    if tx is None:
        print(f"  {label}: VAMP-2 never crosses {PAPER_VAMP2}")
    else:
        print(f"  {label}: VAMP-2 = {PAPER_VAMP2} at τ ≈ {tx:.2f} ns")


# ---- Plot --------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for label, info in RUNS.items():
    pts = results[label]
    taus = [t for t, _ in pts]
    vamps = [v for _, v in pts]
    ax.plot(taus, vamps, marker="o", markersize=3.5, lw=1.4,
            color=info["color"], label=label)

ax.axhline(PAPER_VAMP2, color="black", ls="--", lw=1, alpha=0.7,
           label=f"paper Ghorbani 2022 = {PAPER_VAMP2}")
ax.axhline(V10_TRAIN_BEST, color=RUNS["v10 (with attention)"]["color"],
           ls=":", lw=1, alpha=0.5)
ax.axhline(V4_TRAIN_BEST, color=RUNS["v4 (no attention)"]["color"],
           ls=":", lw=1, alpha=0.5)

for label, info in RUNS.items():
    tx = crossovers[label]
    if tx is not None:
        ax.axvline(tx, color=info["color"], ls=":", lw=0.8, alpha=0.5)
        ax.annotate(f"τ ≈ {tx:.1f} ns", xy=(tx, PAPER_VAMP2),
                    xytext=(8, -14), textcoords="offset points",
                    color=info["color"], fontsize=9)

ax.axvspan(2, 20, alpha=0.07, color="gray", label="user focus zone (2-20 ns)")

ax.axvline(20.0, color="gray", lw=0.6, alpha=0.5)
ax.annotate("trained at τ=20 ns", xy=(20.0, ax.get_ylim()[0] if False else 0),
            xytext=(2, 2), textcoords="offset points",
            color="gray", fontsize=8, alpha=0.7)

ax.set_xscale("log")
ax.set_xlabel("lag time τ (ns)")
ax.set_ylabel("VAMP-2 score (transfer-evaluation)")
ax.set_title("VAMP-2 vs τ — v4 (no attention) vs v10 (with attention)\n"
             "Both trained at τ=20 ns, seed 0; chi computed on full trajectory")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best", framealpha=0.95)

fig.tight_layout()
png_path = OUT_DIR / "villin_tau_scan.png"
fig.savefig(png_path, dpi=140)
print(f"Saved plot: {png_path}")


# ---- Summary printout --------------------------------------------------

print("\n=== Summary ===")
for label in RUNS:
    by_tau = dict(results[label])
    print(f"\n{label}")
    for τ in [2.0, 5.0, 10.0, 15.0, 20.0]:
        if τ in by_tau:
            print(f"  τ={τ:>4.1f} ns: VAMP-2 = {by_tau[τ]:.4f}")
    tx = crossovers[label]
    if tx is not None:
        print(f"  paper 3.78 reached at τ ≈ {tx:.2f} ns")
