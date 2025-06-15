#!/usr/bin/env python3
"""
qm9_histogram.py

Draws simple histograms of QM9 ground-truth distributions for:
  • Dipole moment (target_idx = 0)
  • Internal energy at 0 K, U0 (target_idx = 7)

Plots are saved as a single (10×4) PNG with soft QM9-green styling.
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from edm.qm9 import QM9Dataset
from edm.utils import property_names_safe

# ───────────────────────────────── Styling ───────────────────────────────── #
GREEN_FILL = "#A8D5BA"
GREEN_EDGE = "#5B9279"

# ───────────────────────────────── Helpers ───────────────────────────────── #

def _bin_edges(values: torch.Tensor, resolution: int) -> np.ndarray:
    v = values.cpu().numpy()
    return np.linspace(v.min(), v.max(), resolution + 1)

def _load_raw_qm9_targets(target_idx: int, n_samples: int) -> torch.Tensor:
    dataset = QM9Dataset(
        p=n_samples / 100_000,
        device="cpu",
        batch_size=n_samples,
        atom_scale=1,
        target_idx=0,  # doesn't matter unless you normalise
    )
    dataset.get_data()
    dataloader, _, _ = dataset.make_dataloaders()
    y = dataloader[0].y
    return y[:, target_idx]  # this is the correct fix


# ─────────────────────────────────── Main ────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--save_folder_path", type=str, default="plots/qm9_hist")
    args = parser.parse_args()

    # ───── Load data ───── #
    target_indices: List[int] = [0, 7]  # Dipole moment and U0
    titles = [property_names_safe[i] for i in target_indices]
    # In your main() method
    values = [_load_raw_qm9_targets(i, args.samples) for i in target_indices]

    # ───── Plot histograms ───── #
    sns.set(style="whitegrid")
    Path(args.save_folder_path).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, v, title in zip(axes, values, titles):
        v_np = v.cpu().numpy()
        bins = _bin_edges(v, args.resolution)

        sns.histplot(
            v_np,
            bins=bins,
            stat="count",
            kde=False,
            color=GREEN_FILL,
            edgecolor=GREEN_EDGE,
            linewidth=1.0,
            alpha=0.85,
            ax=ax,
            legend=False,
        )

        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(f"QM9 – {title}")
        if ax.legend_:
            ax.legend_.remove()

    fig.tight_layout()

    outfile = os.path.join(args.save_folder_path, "qm9_hist_dipole_U0.png")
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved histogram to {outfile}")


if __name__ == "__main__":
    main()
