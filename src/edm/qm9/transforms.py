import torch
from ase.data import atomic_masses
from torch_geometric.transforms import BaseTransform


class CenterOfMassTransform(BaseTransform):
    """
    Take in molecule, translate to its center of mass. 
    """
    def __call__(self, data):
        z = data.z.to(torch.long)               # shape [num_atoms]
        masses_np = atomic_masses[z]            # NumPy array, shape [num_atoms]
        # Convert to PyTorch tensor
        masses = torch.tensor(masses_np, device=data.pos.device, dtype=data.pos.dtype)

        total_mass = masses.sum()
        if total_mass > 0:
            com = (masses.unsqueeze(1) * data.pos).sum(dim=0) / total_mass
            data.pos = data.pos - com

        return data
    

class FullyConnectedTransform(BaseTransform):
    """
    Take in molecule, make it fully connected up to range `cutoff`.
    """
    def __init__(self, cutoff=None, eps=1e-2):
        self.cutoff = cutoff
        self.eps = eps

    def __call__(self, data):
        pos = data.pos
        dists = torch.cdist(pos, pos)

        if self.cutoff:
            # If working with PaiNN Property Prediction, we want to process the data
            # such that atoms far away from each other do not have edges
            # This means we re-preprocess samples from EDM before classifying them
            # with a trained PaiNN model
            mask = (dists <= self.cutoff) & (dists > self.eps)
        else:
            mask = (dists > self.eps)

        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        data.edge_index = edge_index
        return data