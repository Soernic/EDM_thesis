import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, random_split
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
from tqdm import tqdm
from edm.qm9.transforms import CenterOfMassTransform, FullyConnectedTransform

from pdb import set_trace

class QM9Dataset:
    def __init__(
            self, 
            p, 
            generator=None, 
            device='cpu', 
            batch_size=100, 
            atom_scale=0.25, # diffusion default
            cutoff_preprocessing=None, # diffusion default, otherwise ~5 Å
            target_idx=None,
            resolution=None
            ):
        
        self.p = p # proportion of dataset
        self.target_idx = target_idx
        self.resolution = resolution
        
        if generator is not None: 
            self.generator = generator
        else: 
            self.generator = torch.Generator().manual_seed(42)

        self.device = device
        self.batch_size = batch_size
        self.atom_scale = atom_scale

        self.center_transform = CenterOfMassTransform()
        self.fc_transform = FullyConnectedTransform(cutoff=cutoff_preprocessing)
        
        self.download_data()
        self.one_hot_converter_setup()

        self.atom_types = {
            1: 0,
            6: 1,
            7: 2,
            8: 3,
            9: 4
        }

        self.atom_types_reverse = torch.tensor([1, 6, 7, 8, 9], device=device)

        # QM9 dataset names we'll use later. 
        self.property_names = {
            0: ('Dipole moment (μ)', 'D'),
            1: ('Isotropic polarizability (α)', 'a₀³'),
            7: ('U₀', 'meV'),
        }

        # From https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/datasets/qm9.html
        self.energy_reference = {
            1:  -13.61312172,   # H
            6: -1029.86312267,  # C
            7: -1485.30251237,  # N
            8: -2042.61123593,  # O
            9: -2713.48485589,  # F
        }

        # from PyG like abvoe
        self.atomrefs = {
            6: [0., 0., 0., 0., 0.],
            7: [
                -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
                -2713.48485589
            ],
            8: [
                -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
                -2713.44632457
            ],
            9: [
                -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
                -2713.42063702
            ],
            10: [
                -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
                -2713.88796536
            ],
            11: [0., 0., 0., 0., 0.],
        }
                

        # Currently only implemented for the U0 energy.
        # self.energy_properties = [7]  # TODO: Add more if there is time so that you can train on all of them
        self.energy_properties = [6, 7, 8, 9, 10, 11]



    def download_data(self):
        self.dataset = QM9(root='./data/QM9')

    def get_data(self):
        assert 0 < self.p <= 1, "p must be in the interval (0, 1])"

        N_full = len(self.dataset)
        subset_size = int(self.p * N_full)

        # EDM split fractions
        train_frac = 100_000 / N_full
        val_frac = 18_000 / N_full

        # random indices
        indices = torch.randperm(N_full, generator=self.generator)[:subset_size]
        dataset_subset = Subset(self.dataset, indices)

        print(f'[info] Preprocessing dataset.. (p = {self.p})')
        dataset_subset = self.process_sequential(dataset_subset)
        
        if self.p == 1: 
            # use EDM splits exactly
            train_len, val_len = 100_000, 18_000
        else:
            train_len = int(round(train_frac * subset_size))
            val_len = int(round(val_frac * subset_size))

        test_len = subset_size - train_len - val_len # fill remainder
        self.train_data, self.val_data, self.test_data = random_split(
            dataset_subset, [train_len, val_len, test_len], generator=self.generator)


    # Taking into account the energy normalisation trick now
    def compute_statistics_and_normalise(self):
        assert self.target_idx is not None, 'target_idx must be set first.'

        is_energy = self.target_idx in self.energy_properties
        print(f'[info] Normalising with '
              f'{"atomic baseline" if is_energy else "min–max"} strategy.')

        if is_energy:
            # subtract per-molecule baseline, no further scaling
            for dataset in (self.train_data, self.val_data, self.test_data):
                for data in dataset:
                    # baseline = data.y.new_tensor(
                    #     [self.energy_reference[int(z)] for z in data.z]
                    # ).sum()

                    
                    ref = torch.as_tensor(self.atomrefs[self.target_idx], device=data.y.device)
                    mapping = self.mapping_tensor.to(data.z.device)
                    baseline = ref[mapping[data.z]].sum()

                    residual = data.y[:, self.target_idx] - baseline
                    data.c = residual                         # convenience copy
                    data.y[:, self.target_idx] = residual      # value seen by the model
            # keep attributes for completeness
            self.mins = None
            self.maxs = None

        else:  # classic min-max on the training set only
            train_targets = torch.cat(
                [d.y[:, self.target_idx] for d in self.train_data]
            )

            self.mins = train_targets.min()
            self.maxs = train_targets.max()
            denom = (self.maxs - self.mins).clamp(min=1e-12)

            for dataset in (self.train_data, self.val_data, self.test_data):
                for data in dataset:
                    normed = (data.y[:, self.target_idx] - self.mins) / denom
                    data.c = normed
                    data.y[:, self.target_idx] = normed

    
    # Now denormalising based on whether it is energy or min-max normalisation going on
    def denormalise(self, y, data_list=None):
        """
        Inverse of `compute_statistics_and_normalise`.

        Parameters
        ----------
        y          : torch.Tensor
            Batched predictions or targets in *normalised* space, shape (N,) or (N,1).
        data_list  : list[torch_geometric.data.Data] | None
            Required for energy targets because we need the atomic numbers
            to rebuild the baseline. Ignored for simple min-max targets.
        """
        is_energy = self.target_idx in self.energy_properties

        if is_energy:
            if data_list is None:
                raise ValueError('`data_list` must be provided for energy targets.')

            y_out = y.clone()
            for i, data in enumerate(data_list):
                # baseline = y.new_tensor(
                #     [self.energy_reference[int(z)] for z in data.z]
                # ).sum()

                ref = torch.as_tensor(self.atomrefs[self.target_idx], device=data.y.device)
                mapping = self.mapping_tensor.to(data.z.device)
                baseline = ref[mapping[data.z]].sum()

                y_out[i] += baseline
                # set_trace()
            return y_out

        # min-max branch
        return y * (self.maxs - self.mins) + self.mins


    # def compute_statistics_and_normalise(self):
    #     if self.target_idx is not None:

    #         # Fill in targets for train_data
    #         all_targets = torch.zeros(len(self.train_data))

    #         for i, data in tqdm(enumerate(self.train_data)):
    #             all_targets[i] = data.y[:, self.target_idx]

    #         self.mins = all_targets.min(dim=0).values
    #         self.maxs = all_targets.max(dim=0).values

    #         print(f'Statistics computed..')
    #         print(f'Minimum: {self.mins:.3f}')
    #         print(f'Maximum: {self.maxs:.3f}')

    #         # Overwrite regression targets
    #         # For train
    #         for i, data in tqdm(enumerate(self.train_data)):
    #             data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mins) / (self.maxs - self.mins)
    #             data.c = data.y[:, self.target_idx] # just duplicate it

    #         # For val
    #         for i, data in tqdm(enumerate(self.val_data)):
    #             data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mins) / (self.maxs - self.mins)
    #             data.c = data.y[:, self.target_idx] # just duplicate it

    #         # For test
    #         for i, data in tqdm(enumerate(self.test_data)):
    #             data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mins) / (self.maxs - self.mins)
    #             data.c = data.y[:, self.target_idx] # just duplicate it
            
    # def denormalise(self, y):
    #     y = y * (self.maxs - self.mins) + self.mins
    #     return y


    def make_dataloaders(self):
        train_loader = self.make_gpu_batches(list(self.train_data), 'train')
        val_loader = self.make_gpu_batches(list(self.val_data), 'val')
        test_loader = self.make_gpu_batches(list(self.test_data), 'test')
        return train_loader, val_loader, test_loader
    
    def process_sequential(self, dataset):
        data_list = [self.apply_transform(data) for data in tqdm(dataset)]
        return data_list

    def apply_transform(self, data):
        data = self.center_transform(data)
        data = self.fc_transform(data)
        data.h = self.one_hot_converter(data.z)
        return data
    
    def make_gpu_batches(self, dataset, label=''):
        return [
            Batch.from_data_list(dataset[i:i+self.batch_size]).to(self.device)
            for i in tqdm(range(0, len(dataset), self.batch_size), desc=f'Preloading batches in {label}')
        ]
    
    def compute_categorical_distribution(self):
        """
        Computes categorical distribution over number of atoms in molecule
        """
        assert hasattr(self, 'train_data'), "Call get_data() before computing p(M)"

        sizes = [data.z.size(0) for data in self.train_data] # atom numbers
        max_M = max(sizes)
        hist = torch.zeros(max_M + 1, dtype=torch.float, device=self.device)

        for m in sizes:
            hist[m] += 1

        probs = hist/hist.sum()
        return probs    
    
    def compute_joint_distribution(self):
        """
        Compute 2D categorical distribution p(c, M) over:
        - c: normalised property value (discretised into bins)
        - M: number of atoms in molecule
        """
        assert hasattr(self, 'train_data'), 'Call get_data() before computing target distribution'
        assert hasattr(self.train_data[0], 'c'), 'Call compute_statistics_and_normalise() before target distribution'

        num_bins = self.resolution

        c_values = []
        M_values = []

        for data in self.train_data:
            c_values.append(data.c.item()) # scalar c in [0, 1] (it has been normalised)
            M_values.append(data.z.size(0)) # number of atoms

        c_values = torch.tensor(c_values, device=self.device)
        M_values = torch.tensor(M_values, device=self.device)

        # Bin c into `num_bins`` intervals
        c_bins = torch.clamp((c_values * num_bins).long(), max=num_bins - 1)
        max_M = M_values.max().item()

        joint_hist = torch.zeros((num_bins, max_M + 1), device=self.device)

        for cb, m in zip(c_bins, M_values):
            joint_hist[cb, m] += 1

        joint_probs = joint_hist / joint_hist.sum()

        return joint_probs



    def one_hot_converter_setup(self):
        self.mapping_tensor = torch.full((10,), -1, dtype=torch.long, device=self.device)
        self.atom_types = torch.tensor([1, 6, 7, 8, 9], device=self.device, dtype=torch.long)
        self.mapping_tensor[self.atom_types] = torch.arange(len(self.atom_types), device=self.device)
    
    def one_hot_converter(self, z):
        mapped_indices = self.mapping_tensor[z]
        h = F.one_hot(mapped_indices, num_classes=len(self.atom_types)).float()
        return h * self.atom_scale


    def build_graph_library(self, max_per_size: int = 1_000):
        """
        Collect at most `max_per_size` edge_index templates for every
        molecule size M that appears in the *training* split.  
        Only edge_index is stored, so the memory footprint is tiny.
        Returns
        -------
        dict[int, list[torch.Tensor]]  mapping M -> [edge_index_i (2xE)]
        """
        import random
        print(f'[preprocessing] Saving {max_per_size} blueprints pr. molecule size')
        lib = {}
        for data in self.train_data:        # already cut-off & centred
            M = data.z.size(0)
            bucket = lib.setdefault(M, [])
            if len(bucket) < max_per_size:
                bucket.append(data.edge_index.clone().cpu())  # keep on CPU
        return lib