import argparse
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from edm.utils.model_utils import load_model, load_prop_pred
from edm.utils import atomic_number_to_symbol, property_names, property_names_safe
from edm.diffusion import EDMSampler
from edm.benchmark import Benchmarks
from edm.qm9 import QM9Dataset
from edm.qm9 import FullyConnectedTransform, CenterOfMassTransform

class HistogramPlotter:
    def __init__(self, resolution, save_folder_path):
        self.resolution = resolution
        sns.set(style='whitegrid')
        os.makedirs(save_folder_path, exist_ok=True)
        self.save_folder_path = save_folder_path

    def plot_hist(self, prop_tensor, name_key, label='QM9'):
        title = property_names_safe[name_key]
        values = prop_tensor.cpu().numpy()
        
        plt.figure(figsize=(8, 4))
        sns.histplot(values, bins=self.resolution, stat='probability', kde=False,
                     color='#A8D5BA', edgecolor='#5B9279', linewidth=1.0)
        plt.title(f'{title} Distribution - {label}', fontsize=14)
        plt.xlabel(f'{title}', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.tight_layout()
        filename = f'{label}_{property_names_safe[name_key]}.png'
        plt.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
        plt.close()

    def plot_double_hist(self, prop_data, prop_edm, name_key, label_data='QM9', label_edm='EDM'):
        title = property_names_safe[name_key]
        values_data = prop_data.cpu().numpy()
        values_edm = prop_edm.cpu().numpy()
        n_data, n_edm = len(values_data), len(values_edm)
        

        # Determine the global min and max across both sets
        min_val = min(values_data.min(), values_edm.min())
        max_val = max(values_data.max(), values_edm.max())

        # Build N+1 equally‐spaced edges from min to max
        bin_edges = np.linspace(min_val, max_val, self.resolution + 1)        

        plt.figure(figsize=(8, 4))
        sns.histplot(values_data, bins=bin_edges, stat='probability', kde=False,
                     color='#A8D5BA', edgecolor='#5B9279', linewidth=1.0,
                     label=f'{label_data} (N={n_data})', alpha=0.7)
        sns.histplot(values_edm, bins=bin_edges, stat='probability', kde=False,
                     color='#AFCBFF', edgecolor='#487EBF', linewidth=1.0,
                     label=f'{label_data} (N={n_edm})', alpha=0.5)
        plt.title(f'{title} Distribution Comparison', fontsize=14)
        plt.xlabel(f'{title}', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.legend()
        plt.tight_layout()
        filename = f'{label_data}_vs_{label_edm}_{property_names_safe[name_key]}.png'
        plt.savefig(os.path.join(self.save_folder_path, filename), dpi=300)
        plt.close()

    def plot_atom_categorical(self, qm9_mols, edm_mols, atom_types=None):
        qm9_atoms = torch.cat([mol.z for mol in qm9_mols], dim=0).cpu().numpy()
        edm_atoms = torch.cat([mol.z for mol in edm_mols], dim=0).cpu().numpy()
        if atom_types is None:
            all_types = sorted(set(qm9_atoms.tolist()) | set(edm_atoms.tolist()))
        else:
            all_types = sorted(atom_types)
        def count_dist(arr):
            return np.array([np.sum(arr == t) for t in all_types])
        qm9_counts, edm_counts = count_dist(qm9_atoms), count_dist(edm_atoms)
        qm9_probs, edm_probs = qm9_counts / qm9_counts.sum(), edm_counts / edm_counts.sum()
        labels = [atomic_number_to_symbol.get(t, str(t)) for t in all_types]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(10, 4))
        plt.bar(x - width/2, qm9_probs, width, label=f'QM9 (N={len(qm9_atoms)})', color='#A8D5BA', edgecolor='#5B9279')
        plt.bar(x + width/2, edm_probs, width, label=f'EDM (N={len(edm_atoms)})', color='#AFCBFF', edgecolor='#487EBF')
        plt.xticks(x, labels)
        plt.xlabel('Atom Type')
        plt.ylabel('Relative Frequency')
        plt.title('Atom Type Distribution (QM9 vs EDM Samples)', fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder_path, 'atom_type_distribution.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edm_path', type=str, required=True)
    parser.add_argument('--prop_pred_paths', nargs='+', required=True)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--resolution', type=int, default=50)
    parser.add_argument('--save_folder_path', type=str, default='plots/histograms')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    plotter = HistogramPlotter(args.resolution, args.save_folder_path)
    edm, noise, edm_cfg = load_model(args.edm_path, args.device)
    sampler = EDMSampler(edm, noise, edm_cfg, args)
    bench = Benchmarks()
    prop_preds = {}
    for path in args.prop_pred_paths:
        model, cfg = load_prop_pred(path, 'cpu')
        model.eval()
        prop_preds[cfg['target_idx']] = (model, cfg)

    samples = sampler.sample(args.samples)
    clean_samples = [mol for mol in samples if bench.run_all([mol], requested=1, q=True)['stability'] == (1.0, 1.0)]
    n = len(clean_samples)

    for target_idx, (model, cfg) in prop_preds.items():
        dataset = QM9Dataset(p=n / 100_000, device='cpu', batch_size=n, atom_scale=1, target_idx=target_idx)
        dataset.get_data()
        dataset.compute_statistics_and_normalise()
        dataloader, _, _ = dataset.make_dataloaders()
        with torch.no_grad():
            data_pred = model(dataloader[0])
            sample_preds = [model(FullyConnectedTransform(cfg['cutoff_data'])(CenterOfMassTransform()(mol))) for mol in clean_samples]
            sample_preds = torch.cat(sample_preds, dim=0)
        plotter.plot_hist(data_pred, name_key=target_idx, label='QM9')
        plotter.plot_double_hist(data_pred, sample_preds, name_key=target_idx)

    plotter.plot_atom_categorical(qm9_mols=dataloader[0].to_data_list(), edm_mols=clean_samples)

if __name__ == '__main__':
    main()
