# src/edm/visualise/edge_retention.py
#!/usr/bin/env python
"""Edge-retention sweep (stability & validity × uniqueness)

This script benchmarks a set of *edge-retention* models whose filenames follow

```
EDM5_XXX.pt   # where XXX ∈ {100, 090, …, 020}
```

For each model we

1. Sample *N* molecules (default **5 000**; change with `--samples`).
2. Call **`Benchmarks.run_all`** once and extract
   * *stability*, and
   * *validity × uniqueness* (i.e. validity-fraction · uniqueness-fraction).
3. Save raw numbers to CSV and create two PNGs with error-bars similar to the
   example you showed.

Run from the project root, e.g.:

```bash
python -m edm.visualise.edge_retention \
       --model_dir models/edge/ \
       --samples 2000
```

Plots end up in `plots/edge/` as
`<nsamp>_<metric>_edge_retention.png`.
"""
import argparse
import os
import re
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from edm.benchmark import Benchmarks
from edm.diffusion import EDMSampler
from edm.utils import load_model

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def extract_retention_pct(fname: str) -> int | None:
    """Return integer edge-retention percentage encoded in *fname* or *None*."""
    m = re.search(r"EDM5_(\d{3})\.pt$", fname)
    return int(m.group(1)) if m else None


def _fraction(value):
    """Normalise helper output: accept either a float or (count, frac) tuple."""
    if isinstance(value, tuple):
        return float(value[1])
    return float(value)


def benchmark_model(path: str, device: str, n: int) -> Tuple[float, float]:
    """Return `(stability, validity×uniqueness)` for *n* samples from *path*."""
    model, noise, cfg = load_model(path, device)
    sampler = EDMSampler(model, noise, cfg, argparse.Namespace(device=device, samples=n))
    mols = sampler.sample(n)

    results: Dict[str, tuple | float] = Benchmarks().run_all(mols, n, q=True)

    stability = _fraction(results["stability"])
    validity  = _fraction(results["validity"])
    unique    = _fraction(results["uniqueness"])

    val_x_uni = validity * unique
    return stability, val_x_uni


def binom_err(p: float, n: int) -> float:
    """Wald standard error for a proportion *p* from *n* trials."""
    return (p * (1 - p) / n) ** 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-retention benchmark sweep")
    ap.add_argument("--model_dir", type=str, default="models/edge/",
                    help="Folder with EDM5_XXX.pt models")
    ap.add_argument("--samples", type=int, default=5000,
                    help="Molecules to sample per model (default 5000)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quiet", action="store_true", help="Suppress info messages")
    args = ap.parse_args()

    sns.set(style="whitegrid")
    out_dir = "plots/edge/"
    os.makedirs(out_dir, exist_ok=True)

    # Lists to accumulate results
    pct_keep:  List[int]   = []
    stability: List[float] = []
    vxu:       List[float] = []  # validity × uniqueness

    for fname in sorted(os.listdir(args.model_dir)):
        if not fname.endswith(".pt"):
            continue
        pct = extract_retention_pct(fname)
        if pct is None:
            continue

        if not args.quiet:
            print(f"[run] {fname} – edge retention {pct}%")
        try:
            stab, prod = benchmark_model(os.path.join(args.model_dir, fname),
                                         args.device, args.samples)
        except Exception as exc:
            print(f"[warn] Skipping {fname}: {exc}")
            continue

        pct_keep.append(pct)
        stability.append(stab)
        vxu.append(prod)

    if not pct_keep:
        print("[error] No models processed – aborting.")
        return

    # Sort by retention percentage for nicer plots
    ordering = sorted(range(len(pct_keep)), key=pct_keep.__getitem__)
    pct_keep = [pct_keep[i] for i in ordering]
    stability = [stability[i] for i in ordering]
    vxu       = [vxu[i]       for i in ordering]

    # Save CSV
    df = pd.DataFrame({
        "edge_retention_pct": pct_keep,
        "stability": stability,
        "valid_x_unique": vxu,
    })
    csv_path = os.path.join(out_dir, f"{args.samples:05}_edge_retention_results.csv")
    df.to_csv(csv_path, index=False)
    if not args.quiet:
        print(f"[done] CSV → {csv_path}")

    # Plot helper
    def _plot(yvals: List[float], ylabel: str, fname_suffix: str):
        errs = [binom_err(p, args.samples) for p in yvals]
        plt.figure(figsize=(10, 4))
        plt.errorbar(pct_keep, [100 * p for p in yvals],
                     yerr=[100 * e for e in errs], fmt="o-", capsize=3,
                     elinewidth=1, ecolor="gray")
        plt.xlabel("Edge retention [%]", fontsize=12)
        plt.ylabel(f"{ylabel} [%]", fontsize=12)
        plt.title(f"{ylabel} vs. Edge retention  ({args.samples} samples/model)", fontsize=12)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{args.samples:05}_{fname_suffix}_edge_retention.png")
        plt.savefig(path)
        if not args.quiet:
            print(f"[done] plot → {path}")
        plt.close()

    # Produce the two plots
    _plot(stability,      "Molecule stability",        "stability")
    _plot(vxu,            "Validity × Uniqueness",     "validxuniq")


if __name__ == "__main__":
    main()