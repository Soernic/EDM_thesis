# graph_connectivity.py
"""
Analyse how many edges survive when QM9 molecules are pruned by a
radius‐cutoff and draw an edge‐retention curve.  The script lives in
`src/edm/visualise/` and is meant to be run from the repository root,
for example:

    python -m edm.visualise.graph_connectivity \
        --p 0.01 --points 50 --max-mols 200

Outputs (PNG + CSV + JSON) are deposited in `plots/qm9/edges/` by
default (the folder is created on demand).

The code integrates with your `QM9Dataset` class in `src/edm/qm9/` and
handles Subset objects correctly.  It samples each molecule once,
computes all pairwise distances on the CPU (cheap for QM9), and uses a
single binary‑search pass per molecule to populate the histogram—so the
runtime scales roughly linearly in `#molecules` rather than in
`#cutoffs × #atoms²`.  You can therefore crank `--points` up to a few
hundred for a silky‑smooth curve without blowing up the job time.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    """Return parsed command‑line options."""
    p = argparse.ArgumentParser(description="Edge‑retention curve on QM9")

    p.add_argument("--p", type=float, default=1.0,
                   help="Fraction of QM9 kept (QM9Dataset argument)")
    p.add_argument("--cutoff-min", type=float, default=0.5,
                   help="Smallest radius in Å")
    p.add_argument("--cutoff-max", type=float, default=8.0,
                   help="Largest radius in Å")
    p.add_argument("--points", type=int, default=100,
                   help="Number of cut‑off points to evaluate")
    p.add_argument("--max-mols", type=int, default=None,
                   help="Limit number of molecules (testing only)")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu or cuda[:N]")
    p.add_argument("--batch", type=int, default=512,
                   help="Unused placeholder to keep old interface stable")
    p.add_argument("--out-dir", type=str, default="plots/qm9/edges",
                   help="Where to save PNG/CSV/JSON outputs")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag appended to output filenames")

    return p.parse_args()

# -----------------------------------------------------------------------------
# Low‑level helpers
# -----------------------------------------------------------------------------

def count_edges_per_cutoff(pos: torch.Tensor, cutoffs: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return (#edges ≤ r for each r,  total possible directed edges)."""
    M: int = pos.size(0)
    E_full: int = M * (M - 1)

    # Pairwise distances (M×M).  Keep on CPU—it is tiny for QM9.
    dists = torch.cdist(pos, pos, p=2.0)
    mask = ~torch.eye(M, dtype=torch.bool)
    d_flat = dists[mask].contiguous()  # (E_full,)
    d_sorted, _ = torch.sort(d_flat)

    counts: torch.Tensor = torch.searchsorted(d_sorted, cutoffs, right=True)
    return counts, E_full

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Deferred import so unit tests / tab‑completion stay fast
    from edm.qm9.data import QM9Dataset  # type: ignore  # noqa: E402

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    ds = QM9Dataset(p=args.p, device=args.device)
    ds.get_data()

    train_subset = ds.train_data  # torch.utils.data.Subset
    # Convert to a concrete list so we can slice easily regardless of Subset/list
    train_mols: List = list(train_subset)
    if args.max_mols is not None:
        train_mols = train_mols[: args.max_mols]

    cutoffs = torch.linspace(args.cutoff_min, args.cutoff_max, args.points)

    tot_possible_edges: int = 0
    tot_edges = torch.zeros_like(cutoffs, dtype=torch.long)

    for data in tqdm(train_mols, desc="Processing", unit="mol"):
        pos = data.pos.cpu()
        counts, E_full = count_edges_per_cutoff(pos, cutoffs)
        tot_possible_edges += E_full
        tot_edges += counts

    pct_retained = (tot_edges.double() / tot_possible_edges) * 100  # [%]

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""

    base = out_dir / f"graph_connectivity{tag}_{date_tag}"

    np.savetxt(base.with_suffix(".csv"),
               np.column_stack([cutoffs.numpy(), pct_retained.numpy()]),
               delimiter=",", header="cutoff_A,pct_edges_retained", comments="")

    with base.with_suffix(".json").open("w") as fp:
        json.dump({"cutoffs": cutoffs.tolist(),
                   "pct_retained": pct_retained.tolist()}, fp)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", palette="crest")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(cutoffs, pct_retained)
    ax.set_xlabel("Cut‑off radius [Å]")
    ax.set_ylabel("Edges retained [%]")
    ax.set_ylim(0, 100)
    ax.set_title("QM9 training set — edge retention vs. cut‑off")
    fig.tight_layout()

    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)

    print(f"[✓] Saved results to {base.with_suffix('.{csv|png|json}')}.parent")


if __name__ == "__main__":
    main()
