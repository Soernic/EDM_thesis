
"""compare_dual_edm_hist.py

Sample molecules from **two** EDM diffusion models, predict chosen
properties with one or more neural-network property predictors and
visualise (i) how each model’s property distribution compares with the
QM9 ground truth (side-by-side histograms) and (ii) how the two models
compare in terms of atom-type frequencies (side-by-side bar charts).
"""

import argparse
import os
from pathlib import Path
from typing import Sequence, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from edm.diffusion import EDMSampler
from edm.qm9 import QM9Dataset, FullyConnectedTransform, CenterOfMassTransform
from edm.utils.model_utils import load_model, load_prop_pred
from edm.utils import property_names_safe, atomic_number_to_symbol
from edm.benchmark import Benchmarks          # ← added

# ────────────────────────────────────────── Plotting helper ────────────────────────────────────────── #

class HistogramPlotter:
    """Render QM9 vs two EDM distributions as 1×2 subplots."""

    def __init__(self, resolution: int, save_folder_path: str):
        self.resolution = resolution
        sns.set(style="whitegrid")
        Path(save_folder_path).mkdir(parents=True, exist_ok=True)
        self.save_folder_path = save_folder_path

    # ---------------------------- continuous property histograms ---------------------------- #

    def _common_edges(self, *arrays: np.ndarray) -> np.ndarray:
        _min = min(arr.min() for arr in arrays)
        _max = max(arr.max() for arr in arrays)
        return np.linspace(_min, _max, self.resolution + 1)

    def plot_hist_two_models(
        self,
        prop_data: torch.Tensor,
        prop_edm_left: torch.Tensor,
        prop_edm_right: torch.Tensor,
        name_key: int,
        label_data: str = "QM9",
        edm_labels: Tuple[str, str] = ("linear", "cosine"),
    ) -> None:
        """Save a figure with two subplots (QM9 vs each EDM)."""
        title = property_names_safe[name_key]
        v_data = prop_data.cpu().numpy()
        v_left = prop_edm_left.cpu().numpy()
        v_right = prop_edm_right.cpu().numpy()

        # Shared bin edges so the two subplots are directly comparable
        bin_edges = self._common_edges(v_data, v_left, v_right)

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, vals, lbl in zip(axes, [v_left, v_right], edm_labels):
            sns.histplot(
                v_data,
                bins=bin_edges,
                stat="probability",
                kde=False,
                color="#A8D5BA",
                edgecolor="#5B9279",
                linewidth=1.0,
                label=f"{label_data} (N={len(v_data)})",
                alpha=0.8,
                ax=ax,
            )
            sns.histplot(
                vals,
                bins=bin_edges,
                stat="probability",
                kde=False,
                color="#AFCBFF",
                edgecolor="#487EBF",
                linewidth=1.0,
                label=f"{lbl} (N={len(vals)})",
                alpha=0.4,
                ax=ax,
            )
            ax.set_xlabel(f"Normalised {title}")
            ax.set_title(f"{label_data} vs {lbl}")
            ax.legend()
        axes[0].set_ylabel("Frequency")
        fig.tight_layout()

        filename = (
            f"QM9_vs_{edm_labels[0]}_vs_{edm_labels[1]}_{property_names_safe[name_key]}.png"
        )
        fig.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
        plt.close(fig)

    # ------------------------------- categorical atom types -------------------------------- #

    def plot_atom_type_two_models(
        self,
        qm9_mols: Sequence,
        mols_left: Sequence,
        mols_right: Sequence,
        edm_labels: Tuple[str, str] = ("linear", "cosine"),
        label_data: str = "QM9",
    ) -> None:
        """Bar‑chart QM9 vs each EDM model (1×2 subplots)."""
        # Gather atomic‑number arrays
        def stack_z(mols: Sequence) -> np.ndarray:
            return (
                torch.cat([mol.z for mol in mols], dim=0).cpu().numpy()
                if len(mols) > 0
                else np.array([])
            )

        z_qm9 = stack_z(qm9_mols)
        z_left = stack_z(mols_left)
        z_right = stack_z(mols_right)

        # Universe of atom types across all three sets
        atom_types = sorted(set(z_qm9.tolist()) | set(z_left.tolist()) | set(z_right.tolist()))

        def freq(arr):
            counts = np.array([np.sum(arr == t) for t in atom_types])
            total = counts.sum() or 1  # avoid div‑by‑zero
            return counts / total

        freq_data = freq(z_qm9)
        freq_left = freq(z_left)
        freq_right = freq(z_right)

        labels = [atomic_number_to_symbol.get(t, str(t)) for t in atom_types]
        x = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, probs, lbl, z_model in zip(
            axes,
            [freq_left, freq_right],
            edm_labels,
            [z_left, z_right],
        ):
            ax.bar(
                x - width / 2,
                freq_data,
                width,
                label=f"{label_data} (N={len(z_qm9)})",
                color="#A8D5BA",
                edgecolor="#5B9279",
            )
            ax.bar(
                x + width / 2,
                probs,
                width,
                label=f"{lbl} (N={len(z_model)})",
                color="#AFCBFF",
                edgecolor="#487EBF",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Atom type")
            ax.set_title(f"{label_data} vs {lbl}")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Relative frequency")
        fig.tight_layout()

        filename = f"QM9_vs_{edm_labels[0]}_vs_{edm_labels[1]}_atom_type_distribution.png"
        fig.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
        plt.close(fig)

# ────────────────────────────────────────── Helper functions ────────────────────────────────────────── #
def get_predictions_for_mols(
    mols: Sequence,
    model: torch.nn.Module,
    cutoff: float,
) -> torch.Tensor:
    """Apply a property-prediction network to a list of molecule objects."""
    with torch.no_grad():
        preds = [
            model(FullyConnectedTransform(cutoff)(CenterOfMassTransform()(mol)))
            for mol in mols
        ]
    return torch.cat(preds, dim=0) if preds else torch.empty(0)

# ────────────────────────────────────────── Main program ───────────────────────────────────────────── #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edm_paths", nargs=2, required=True, metavar=("LINEAR", "COSINE"))
    parser.add_argument("--edm_labels", nargs=2, default=["linear", "cosine"], metavar=("LABEL_LEFT", "LABEL_RIGHT"))
    parser.add_argument("--prop_pred_paths", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--save_folder_path", type=str, default="plots/histograms_two_edms")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ------------------------------- Load EDM samplers ------------------------------- #
    samplers: List[EDMSampler] = []
    for path in args.edm_paths:
        edm, noise, edm_cfg = load_model(path, args.device)
        samplers.append(EDMSampler(edm, noise, edm_cfg, args))

    # ------------------------------- Property nets ----------------------------------- #
    prop_preds = {}
    for path in args.prop_pred_paths:
        model, cfg = load_prop_pred(path, device="cpu")
        model.eval()
        prop_preds[cfg["target_idx"]] = (model, cfg)

    # -------------------------------- Sampling --------------------------------------- #
    raw_sample_sets = [sampler.sample(args.samples) for sampler in samplers]

    # --------------------------- Stability filtering (from v1) ----------------------- #
    bench = Benchmarks()
    stable_sets: List[List] = []
    for mols in raw_sample_sets:
        stable = [
            mol
            for mol in mols
            if bench.run_all([mol], requested=1, q=True)["stability"] == (1.0, 1.0)
        ]
        stable_sets.append(stable)

    n_stable = min(len(s) for s in stable_sets)
    if n_stable == 0:
        raise RuntimeError("No stable molecules were found in at least one EDM sample set.")

    # Trim each model’s list so both have the same N
    sample_sets = [s[:n_stable] for s in stable_sets]

    # ------------------------------- Visual output ----------------------------------- #
    plotter = HistogramPlotter(args.resolution, args.save_folder_path)

    # We will capture QM9 molecules from the first property loop to reuse for atom-type plot
    qm9_mols: Sequence = []

    for target_idx, (model, cfg) in prop_preds.items():
        # ---- Ground-truth QM9 subset ---- #
        dataset = QM9Dataset(
            p=n_stable / 100_000,          # match N to stable sample count
            device="cpu",
            batch_size=n_stable,
            atom_scale=1,
            target_idx=target_idx,
        )
        dataset.get_data()
        dataset.compute_statistics_and_normalise()
        dataloader, _, _ = dataset.make_dataloaders()
        if not qm9_mols:
            qm9_mols = dataloader[0].to_data_list()

        with torch.no_grad():
            data_pred: torch.Tensor = model(dataloader[0])

        # ---- Predictions for each EDM sample set ---- #
        preds_left  = get_predictions_for_mols(sample_sets[0], model, cfg["cutoff_data"])
        preds_right = get_predictions_for_mols(sample_sets[1], model, cfg["cutoff_data"])

        # ---- Draw and save histogram ---- #
        plotter.plot_hist_two_models(
            data_pred,
            preds_left,
            preds_right,
            name_key=target_idx,
            edm_labels=tuple(args.edm_labels),
        )

    # ----------------------- Atom-type categorical distribution ----------------------- #
    if qm9_mols:
        plotter.plot_atom_type_two_models(
            qm9_mols,
            sample_sets[0],
            sample_sets[1],
            edm_labels=tuple(args.edm_labels),
        )

if __name__ == "__main__":
    main()


























# """compare_dual_edm_hist.py

# Sample molecules from **two** EDM diffusion models, predict chosen
# properties with one or more neural‑network property predictors and
# visualise (i) how each model’s property distribution compares with the
# QM9 ground truth (side‑by‑side histograms) and (ii) how the two models
# compare in terms of atom‑type frequencies (side‑by‑side bar charts).

# Running example:

# python compare_dual_edm_hist.py \
#     --edm_paths path/to/linear.pt path/to/cosine.pt \
#     --prop_pred_paths models/alpha.pt models/beta.pt \
#     --samples 2000 \
#     --save_folder_path plots/my_run

# The two EDMs are labelled **linear** and **cosine** by default; you can
# override this with --edm_labels.
# """

# import argparse
# import os
# from pathlib import Path
# from typing import Sequence, Tuple, List

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import torch

# from edm.diffusion import EDMSampler
# from edm.qm9 import QM9Dataset, FullyConnectedTransform, CenterOfMassTransform
# from edm.utils.model_utils import load_model, load_prop_pred
# from edm.utils import property_names_safe, atomic_number_to_symbol

# # ────────────────────────────────────────── Plotting helper ────────────────────────────────────────── #

# class HistogramPlotter:
#     """Render QM9 vs two EDM distributions as 1×2 subplots."""

#     def __init__(self, resolution: int, save_folder_path: str):
#         self.resolution = resolution
#         sns.set(style="whitegrid")
#         Path(save_folder_path).mkdir(parents=True, exist_ok=True)
#         self.save_folder_path = save_folder_path

#     # ---------------------------- continuous property histograms ---------------------------- #

#     def _common_edges(self, *arrays: np.ndarray) -> np.ndarray:
#         _min = min(arr.min() for arr in arrays)
#         _max = max(arr.max() for arr in arrays)
#         return np.linspace(_min, _max, self.resolution + 1)

#     def plot_hist_two_models(
#         self,
#         prop_data: torch.Tensor,
#         prop_edm_left: torch.Tensor,
#         prop_edm_right: torch.Tensor,
#         name_key: int,
#         label_data: str = "QM9",
#         edm_labels: Tuple[str, str] = ("linear", "cosine"),
#     ) -> None:
#         """Save a figure with two subplots (QM9 vs each EDM)."""
#         title = property_names_safe[name_key]
#         v_data = prop_data.cpu().numpy()
#         v_left = prop_edm_left.cpu().numpy()
#         v_right = prop_edm_right.cpu().numpy()

#         # Shared bin edges so the two subplots are directly comparable
#         bin_edges = self._common_edges(v_data, v_left, v_right)

#         fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
#         for ax, vals, lbl in zip(axes, [v_left, v_right], edm_labels):
#             sns.histplot(
#                 v_data,
#                 bins=bin_edges,
#                 stat="probability",
#                 kde=False,
#                 color="#A8D5BA",
#                 edgecolor="#5B9279",
#                 linewidth=1.0,
#                 label=f"{label_data} (N={len(v_data)})",
#                 alpha=0.8,
#                 ax=ax,
#             )
#             sns.histplot(
#                 vals,
#                 bins=bin_edges,
#                 stat="probability",
#                 kde=False,
#                 color="#AFCBFF",
#                 edgecolor="#487EBF",
#                 linewidth=1.0,
#                 label=f"{lbl} (N={len(vals)})",
#                 alpha=0.4,
#                 ax=ax,
#             )
#             ax.set_xlabel(f"Normalised {title}")
#             ax.set_title(f"{label_data} vs {lbl}")
#             ax.legend()
#         axes[0].set_ylabel("Frequency")
#         fig.tight_layout()

#         filename = (
#             f"QM9_vs_{edm_labels[0]}_vs_{edm_labels[1]}_{property_names_safe[name_key]}.png"
#         )
#         fig.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
#         plt.close(fig)

#     # ------------------------------- categorical atom types -------------------------------- #

#     def plot_atom_type_two_models(
#         self,
#         qm9_mols: Sequence,
#         mols_left: Sequence,
#         mols_right: Sequence,
#         edm_labels: Tuple[str, str] = ("linear", "cosine"),
#         label_data: str = "QM9",
#     ) -> None:
#         """Bar‑chart QM9 vs each EDM model (1×2 subplots)."""
#         # Gather atomic‑number arrays
#         def stack_z(mols: Sequence) -> np.ndarray:
#             return (
#                 torch.cat([mol.z for mol in mols], dim=0).cpu().numpy()
#                 if len(mols) > 0
#                 else np.array([])
#             )

#         z_qm9 = stack_z(qm9_mols)
#         z_left = stack_z(mols_left)
#         z_right = stack_z(mols_right)

#         # Universe of atom types across all three sets
#         atom_types = sorted(set(z_qm9.tolist()) | set(z_left.tolist()) | set(z_right.tolist()))

#         def freq(arr):
#             counts = np.array([np.sum(arr == t) for t in atom_types])
#             total = counts.sum() or 1  # avoid div‑by‑zero
#             return counts / total

#         freq_data = freq(z_qm9)
#         freq_left = freq(z_left)
#         freq_right = freq(z_right)

#         labels = [atomic_number_to_symbol.get(t, str(t)) for t in atom_types]
#         x = np.arange(len(labels))
#         width = 0.35

#         fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
#         for ax, probs, lbl, z_model in zip(
#             axes,
#             [freq_left, freq_right],
#             edm_labels,
#             [z_left, z_right],
#         ):
#             ax.bar(
#                 x - width / 2,
#                 freq_data,
#                 width,
#                 label=f"{label_data} (N={len(z_qm9)})",
#                 color="#A8D5BA",
#                 edgecolor="#5B9279",
#             )
#             ax.bar(
#                 x + width / 2,
#                 probs,
#                 width,
#                 label=f"{lbl} (N={len(z_model)})",
#                 color="#AFCBFF",
#                 edgecolor="#487EBF",
#             )
#             ax.set_xticks(x)
#             ax.set_xticklabels(labels)
#             ax.set_xlabel("Atom type")
#             ax.set_title(f"{label_data} vs {lbl}")
#             ax.legend(fontsize=8)
#         axes[0].set_ylabel("Relative frequency")
#         fig.tight_layout()

#         filename = f"QM9_vs_{edm_labels[0]}_vs_{edm_labels[1]}_atom_type_distribution.png"
#         fig.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
#         plt.close(fig)

# # ────────────────────────────────────────── Helper functions ────────────────────────────────────────── #

# def get_predictions_for_mols(
#     mols: Sequence,
#     model: torch.nn.Module,
#     cutoff: float,
# ) -> torch.Tensor:
#     """Apply a property‑prediction network to a list of molecule objects."""
#     with torch.no_grad():
#         preds = [
#             model(FullyConnectedTransform(cutoff)(CenterOfMassTransform()(mol)))
#             for mol in mols
#         ]
#     return torch.cat(preds, dim=0) if preds else torch.empty(0)

# # ────────────────────────────────────────── Main program ───────────────────────────────────────────── #

# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--edm_paths",
#         nargs=2,
#         required=True,
#         metavar=("LINEAR", "COSINE"),
#         help="Paths to two EDM checkpoints.",
#     )
#     parser.add_argument(
#         "--edm_labels",
#         nargs=2,
#         default=["linear", "cosine"],
#         metavar=("LABEL_LEFT", "LABEL_RIGHT"),
#         help="Readable labels for the two EDM models.",
#     )
#     parser.add_argument("--prop_pred_paths", nargs="+", required=True)
#     parser.add_argument("--samples", type=int, default=1000)
#     parser.add_argument("--resolution", type=int, default=50)
#     parser.add_argument(
#         "--save_folder_path",
#         type=str,
#         default="plots/histograms_two_edms",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#     )
#     args = parser.parse_args()

#     # ------------------------------- Load EDM samplers ------------------------------- #
#     samplers: List[EDMSampler] = []
#     for path in args.edm_paths:
#         edm, noise, edm_cfg = load_model(path, args.device)
#         samplers.append(EDMSampler(edm, noise, edm_cfg, args))

#     # ------------------------------- Property nets ----------------------------------- #
#     prop_preds = {}
#     for path in args.prop_pred_paths:
#         model, cfg = load_prop_pred(path, device="cpu")
#         model.eval()
#         prop_preds[cfg["target_idx"]] = (model, cfg)

#     # -------------------------------- Sampling --------------------------------------- #
#     sample_sets = [sampler.sample(args.samples) for sampler in samplers]

#     # ------------------------------- Visual output ----------------------------------- #
#     plotter = HistogramPlotter(args.resolution, args.save_folder_path)

#     # We will capture QM9 molecules from the first property loop to reuse for atom‑type plot
#     qm9_mols: Sequence = []

#     for target_idx, (model, cfg) in prop_preds.items():
#         # ---- Ground‑truth QM9 subset ---- #
#         dataset = QM9Dataset(
#             p=args.samples / 100_000,
#             device="cpu",
#             batch_size=args.samples,
#             atom_scale=1,
#             target_idx=target_idx,
#         )
#         dataset.get_data()
#         dataset.compute_statistics_and_normalise()
#         dataloader, _, _ = dataset.make_dataloaders()
#         if not qm9_mols:
#             # Save for atom‑type comparison later
#             qm9_mols = dataloader[0].to_data_list()

#         with torch.no_grad():
#             data_pred: torch.Tensor = model(dataloader[0])

#         # ---- Predictions for each EDM sample set ---- #
#         preds_left = get_predictions_for_mols(sample_sets[0], model, cfg["cutoff_data"])
#         preds_right = get_predictions_for_mols(sample_sets[1], model, cfg["cutoff_data"])

#         # ---- Draw and save histogram ---- #
#         plotter.plot_hist_two_models(
#             data_pred,
#             preds_left,
#             preds_right,
#             name_key=target_idx,
#             edm_labels=tuple(args.edm_labels),
#         )

#     # ----------------------- Atom‑type categorical distribution ----------------------- #
#     if qm9_mols:
#         plotter.plot_atom_type_two_models(
#             qm9_mols,
#             sample_sets[0],
#             sample_sets[1],
#             edm_labels=tuple(args.edm_labels),
#         )

# if __name__ == "__main__":
#     main()

