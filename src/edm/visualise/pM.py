# pM.py — Visualise p(M) for QM9
"""
Draw the empirical distribution of molecule sizes (number of atoms *M*)
for the QM9 training split.  The script lives in
`src/edm/visualise/` and follows the same command‑line style as
`graph_connectivity.py`.

Typical quick run:

    python -m edm.visualise.pM --p 0.02 --max-mols 5000

A full job might be:

    python -m edm.visualise.pM --p 1 --out-dir plots/qm9/atoms

Outputs: a CSV (M, count, probability), a JSON copy, and a high‑dpi PNG
bar‑plot using Seaborn’s *crest* palette.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Histogram of QM9 atom counts p(M)")

    p.add_argument("--p", type=float, default=1.0,
                   help="Fraction of QM9 kept (QM9Dataset argument)")
    p.add_argument("--max-mols", type=int, default=None,
                   help="Limit number of molecules (testing only)")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu or cuda[:N]")
    p.add_argument("--out-dir", type=str, default="plots/qm9/atoms",
                   help="Where to save PNG/CSV/JSON outputs")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag appended to output filenames")

    return p.parse_args()

# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from edm.qm9.data import QM9Dataset  # type: ignore  # noqa: E402

    ds = QM9Dataset(p=args.p, device=args.device)
    ds.get_data()

    train_subset = ds.train_data  # torch.utils.data.Subset
    train_mols: List = list(train_subset)
    if args.max_mols is not None:
        train_mols = train_mols[: args.max_mols]

    sizes = [data.z.size(0) for data in tqdm(train_mols, desc="Scanning", unit="mol")]
    sizes_arr = np.array(sizes, dtype=np.int64)

    max_M = sizes_arr.max()
    counts = np.bincount(sizes_arr, minlength=max_M + 1)
    probs = counts / counts.sum()

    Ms = np.arange(max_M + 1)
    nonzero = counts > 0
    Ms = Ms[nonzero]
    counts = counts[nonzero]
    probs = probs[nonzero]

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""

    base = out_dir / f"pM{tag}_{date_tag}"

    np.savetxt(base.with_suffix(".csv"),
               np.column_stack([Ms, counts, probs]),
               delimiter=",", header="M,count,probability", comments="")

    with base.with_suffix(".json").open("w") as fp:
        json.dump({"M": Ms.tolist(), "count": counts.tolist(), "prob": probs.tolist()}, fp)

    # ------------------------------------------------------------------
    # Plot p(M) with Matplotlib
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    bar_color = "#A8D5BA"  # soft light blue
    edge_color = "#5B9279"  # soft light blue
    ax.bar(Ms, probs, color=bar_color, width=0.8, edgecolor=edge_color, linewidth=1.5)

    ax.set_title("Distribution of Number of Atoms per Molecule (p(M))", fontsize=14)
    ax.set_xlabel("Number of atoms (M)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xticks(Ms)  # ensure ticks are centered on bars
    ax.set_xlim(Ms.min() - 0.5, Ms.max() + 0.5)

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close()





    # # Define minimum value to start x-axis at (QM9 has molecules with M = 5 to 29)
    # min_plot_M = 0
    # if Ms.min() > min_plot_M:
    #     pad_width = Ms.min() - min_plot_M
    #     Ms = np.concatenate([np.arange(min_plot_M, Ms.min()), Ms])
    #     counts = np.concatenate([np.zeros(pad_width, dtype=int), counts])

    # # Set style and color
    # sns.set_theme(style="whitegrid")
    # soft_green = mcolors.to_rgba("#b2df8a")  # soft green from colorbrewer

    # # Create figure
    # fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    # sns.barplot(x=Ms, y=counts, ax=ax, color=soft_green, width=0.8)

    # # Labels and title
    # ax.set_xlabel("Number of atoms per molecule, M", labelpad=10)
    # ax.set_ylabel("Count (training split)", labelpad=10)
    # ax.set_title("QM9 training set — Distribution of Atom Counts p(M)", pad=15)

    # # X-axis range and ticks
    # ax.set_xlim(min_plot_M - 0.5, Ms.max() + 0.5)
    # ax.set_xticks(np.arange(min_plot_M, Ms.max() + 1))

    # # Layout and save
    # fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    # fig.savefig(base.with_suffix(".png"))
    # plt.close(fig)

    print(f"\n[✓] Saved p(M) histogram to {base.with_suffix('.{csv|png|json}')}.parent")


if __name__ == "__main__":
    main()
