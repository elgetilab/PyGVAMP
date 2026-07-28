"""Check ab42 all-phase val VAMP-2 against the k=4 theoretical ceiling (4.0).

VAMP-2 = 1 + sum_{i=2..k} sigma_i^2 with sigma_i <= 1, so k=4 => max 4.0.
Any epoch above that is an estimator artifact, not a better model.
"""
import re
import statistics as st
from pathlib import Path

PAT = re.compile(
    r"\[(chi|us|all)\] epoch (\d+): val VAMP-2=([0-9.]+)\s+VAMP-E=([0-9.]+)")
INIT = re.compile(r"algebraic U/S init done .*val VAMP-2=([0-9.]+)\s+VAMP-E=([0-9.]+)")

rows = []
for seed in range(10):
    f = Path(f"/mnt/hdd/experiments/logs/ab42_rev_838_{seed}.out")
    txt = f.read_text(errors="ignore")
    alls = [(int(m.group(2)), float(m.group(3)))
            for m in PAT.finditer(txt) if m.group(1) == "all"]
    init = INIT.search(txt)
    init_v2 = float(init.group(1)) if init else float("nan")
    vals = [v for _, v in alls]
    over = [v for v in vals if v > 4.0]
    tail = vals[-10:]
    rows.append(dict(
        seed=seed, n=len(vals), init=init_v2,
        mx=max(vals), median=st.median(vals),
        tail_med=st.median(tail), n_over=len(over),
        max_over=max(over) if over else 0.0))

print(f"{'seed':>4} {'n_ep':>5} {'init':>7} {'median':>8} {'last10med':>10} "
      f"{'MAX(sel)':>9} {'#>4.0':>6} {'excess':>7}")
for r in rows:
    print(f"{r['seed']:>4} {r['n']:>5} {r['init']:>7.4f} {r['median']:>8.4f} "
          f"{r['tail_med']:>10.4f} {r['mx']:>9.4f} {r['n_over']:>6} "
          f"{max(0.0, r['mx'] - 4.0):>7.4f}")

med = [r['tail_med'] for r in rows]
mx = [r['mx'] for r in rows]
print()
print(f"Selected (max-over-epoch) : {st.mean(mx):.4f} +- {st.stdev(mx):.4f}")
print(f"Converged (last-10 median): {st.mean(med):.4f} +- {st.stdev(med):.4f}")
print(f"Paper                     : 3.99 +- 0.002   (k=4 ceiling = 4.0)")
print(f"seeds with >=1 epoch above ceiling: {sum(1 for r in rows if r['n_over'])}/10")
print(f"total selection epochs above ceiling: {sum(r['n_over'] for r in rows)}"
      f" / {sum(r['n'] for r in rows)}")
