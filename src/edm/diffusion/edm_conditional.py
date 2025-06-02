import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool
from edm.diffusion import LinearNoiseSchedule, CosineNoiseSchedule, EDM
from edm.models import PaiNNDiffusion, PaiNNConditionalDiffusion


class ConditionalEDM(EDM):
    """
    For conditional EDM very little changes. We use a different PaiNN model that 
    in the forward pass uses the property data.y as an extra context layer, 
    passing whichever regression target we choose through a sinusoidal embedding
    and an MLP just like the time embedding. It is very similar. And, since this 
    is all in the data object anyways, the rest of the method stays totally the same. 
    
    The only change we'll need is a different sampling method, which should add a 
    regression target to the data object, and that target should importantly be in 
    the same index as it originally was

    ... or we do something else. Is data.c taken?
    """
    def __init__(
            self,
            noise_schedule,
            num_rounds=9,
            state_dim=192,
            cutoff_painn=5,
            edge_dim=64,
            T=1000,
            target_idx=1 # alpha / polarizability tensor
    ):
        super().__init__(noise_schedule, num_rounds, state_dim, cutoff_painn, edge_dim, T)
        self.target_idx = target_idx

        self.model = PaiNNConditionalDiffusion(
            num_rounds,
            state_dim,
            cutoff_painn,
            edge_dim,
            target_idx # ??
        )


