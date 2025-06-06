import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from pdb import set_trace

from edm.utils.model_utils import load_model, load_prop_pred
from edm.utils import atomic_number_to_symbol
from edm.utils import property_names_safe as property_names
from edm.diffusion import EDMSampler
from edm.benchmark import Benchmarks
from edm.qm9 import QM9Dataset
from edm.qm9 import FullyConnectedTransform, CenterOfMassTransform

class HistogramPlotter:
    """
    Plots one or two histograms of predicted molecular properties.
    """

    def __init__(self, resolution, save_folder_path):
        self.resolution = resolution
        sns.set(style='whitegrid')
        os.makedirs(save_folder_path, exist_ok=True)
        self.save_folder_path = save_folder_path

    def plot_hist(self, prop_tensor, target_idx, label='QM9'):
        values = prop_tensor.cpu().numpy()
        title, unit = property_names[target_idx]

        plt.figure(figsize=(8, 4))
        ax = sns.histplot(
            values,
            bins=self.resolution,
            stat='probability',
            kde=False,
            color='#A8D5BA',
            edgecolor='#5B9279',
            linewidth=1.0,
        )
        ax.set_title(f'{title} Distribution - {label}', fontsize=14)
        ax.set_xlabel(f'{title} [{unit}]', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)

        plt.tight_layout()
        filename = f'{label}_{title.replace(" ", "_")}.png'
        save_path = os.path.join(self.save_folder_path, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_double_hist(self, prop_data, prop_edm, target_idx, label_data='QM9', label_edm='EDM'):
        values_data = prop_data.cpu().numpy()
        values_edm = prop_edm.cpu().numpy()
        title, unit = property_names[target_idx]

        plt.figure(figsize=(8, 4))
        ax = sns.histplot(
            values_data,
            bins=self.resolution,
            stat='probability',
            kde=False,
            color='#A8D5BA',
            edgecolor='#5B9279',
            linewidth=1.0,
            label=label_data,
            alpha=0.7,
        )
        sns.histplot(
            values_edm,
            bins=self.resolution,
            stat='probability',
            kde=False,
            color='#AFCBFF',
            edgecolor='#487EBF',
            linewidth=1.0,
            label=label_edm,
            alpha=0.5,
        )

        ax.set_title(f'{title} Distribution Comparison', fontsize=14)
        ax.set_xlabel(f'{title} [{unit}]', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.legend()

        plt.tight_layout()
        filename = f'{label_data}_vs_{label_edm}_{title.replace(" ", "_")}.png'
        save_path = os.path.join(self.save_folder_path, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()


    def plot_atom_categorical(self, qm9_mols, edm_mols, atom_types=None):
        """
        Plots overlaid categorical distributions of atom types for QM9 and EDM molecules.
        `atom_types` should be a list of atomic numbers to include in the plot (optional).
        """

        # Flatten all atomic numbers
        qm9_atoms = torch.cat([mol.z for mol in qm9_mols], dim=0).cpu().numpy()
        edm_atoms = torch.cat([mol.z for mol in edm_mols], dim=0).cpu().numpy()

        # Determine all atom types to consider
        if atom_types is None:
            all_types = sorted(set(qm9_atoms.tolist()) | set(edm_atoms.tolist()))
        else:
            all_types = sorted(atom_types)

        # Count and normalize
        def count_dist(atom_array):
            return np.array([np.sum(atom_array == t) for t in all_types])

        qm9_counts = count_dist(qm9_atoms)
        edm_counts = count_dist(edm_atoms)

        qm9_probs = qm9_counts / qm9_counts.sum()
        edm_probs = edm_counts / edm_counts.sum()

        labels = [atomic_number_to_symbol.get(t, str(t)) for t in all_types]

        # Plotting
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 4))
        plt.bar(x - width/2, qm9_probs, width, label=f'QM9 (N={len(qm9_atoms)})', color='#A8D5BA', edgecolor='#5B9279')
        plt.bar(x + width/2, edm_probs, width, label=f'EDM (N={len(edm_atoms)})', color='#AFCBFF', edgecolor='#487EBF')

        plt.xticks(x, labels)
        plt.xlabel('Atomic Number (Z)')
        plt.ylabel('Relative Frequency')
        plt.title('Atom Type Distribution (QM9 vs EDM Samples)', fontsize=13)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.save_folder_path, 'atom_type_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise histogram of properties on molecules, from dataset or generated'
    )

    parser.add_argument('--prop_pred_path', type=str, default='models/mu.pt', help='Path to property prediction model')
    parser.add_argument('--edm_path', type=str, default='models/edm.pt', help='Path to EDM sampler model')

    parser.add_argument('--samples', type=int, default=1000, help='How many samples to generate with EDM')
    parser.add_argument('--resolution', type=int, default=50, help='How many bins there should be in the histogram')

    parser.add_argument('--save_folder_path', type=str, default='plots/histograms', help='where to save plots')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for sampling')

    return parser.parse_args()


if __name__ == '__main__':
    """
    Load in property prediction model from 
    """

    args = parse_args()
    plotter = HistogramPlotter(resolution=args.resolution, save_folder_path=args.save_folder_path)

    prop_pred, prop_pred_cfg = load_prop_pred(args.prop_pred_path, 'cpu') # since samples will be on cpu
    edm, noise, edm_cfg = load_model(args.edm_path, args.device)
    
    # Set to eval mode
    prop_pred.eval()
    edm.eval()

    sampler = EDMSampler(edm, noise, edm_cfg, args)
    bench = Benchmarks() # for EDM stability checks
    fc_transform = FullyConnectedTransform(cutoff=prop_pred_cfg['cutoff_data'])
    com_transform = CenterOfMassTransform()

    # Grab samples from EDM model
    samples = sampler.sample(args.samples) 
    clean_samples = []

    # Check for stability and validity
    for idx, mol in enumerate(samples):
        res = bench.run_all([mol], requested=1, q=True)
        if res['stability'] == (1.0, 1.0) and res['validity'] == 1.0:
            clean_samples.append(mol)

    # This is how many we're gonna grab from the QM9 Dataset
    number_of_samples = len(clean_samples)

    qm9 = QM9Dataset(
        p=number_of_samples / 100_000, 
        device='cpu', # since samples will be on cpu
        batch_size=number_of_samples, 
        atom_scale=1, 
        target_idx=prop_pred_cfg['target_idx']
        )
    
    qm9.get_data()
    qm9.compute_statistics_and_normalise()
    dataloader, _, _ = qm9.make_dataloaders()
    
    # Predictions
    with torch.no_grad():

        # Data predictions
        # NOTE: Using predictions of target, rather than true target for fair comparison if there 
        # is bias in the predictor
        data_pred = prop_pred(dataloader[0]) # since they are all batched up

        # Sample predictions
        sample_preds = []
        for idx, data in enumerate(clean_samples):
            
            # Preprocess to add edge index
            transformed_data = fc_transform(com_transform(data))

            # Make predictions
            sample_preds.append(prop_pred(transformed_data))

        sample_preds = torch.cat(sample_preds, dim=0)

    target_idx = prop_pred_cfg['target_idx']
    plotter.plot_hist(data_pred, target_idx=target_idx, label='QM9')
    plotter.plot_double_hist(data_pred, sample_preds, target_idx=target_idx)
    plotter.plot_atom_categorical(qm9_mols=dataloader[0].to_data_list(), edm_mols=clean_samples)
