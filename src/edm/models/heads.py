import torch
import torch.nn as nn


class InvariantReadout(nn.Module):
    """
    Takes in scalar (s) states and passes them through an MLP, reducing them to 
    a dimension out_dim. Used for invariant predictions such as energy or 
    atom type one-hot vectors.
    """
    def __init__(self, state_dim: int, out_dim: int = 5):
        super().__init__()
        
        self.readout = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, out_dim) # number of elements
        )

    def forward(self, s):
        return self.readout(s)


class EquivariantReadout(nn.Module):
    """
    Takes in both scalar (s) and vector (v) states and combine them in an 
    equivariance preserving way to produce a set of atomwise vector 
    states used for, e.g., prediciton of mu or alpha, or the noise 
    on the positions in a diffusion process. 

    Note: Forward process produces atomwise outputs, it does not sum
    over the molecule. That happens in the main neural net for flexibility
    reasons. 

    Parameters
    ----------
    state_dim : int
        Feature size F of both s  [N, F]  and v  [N, F, 3].
    """
    def __init__(self, state_dim: int):
        super().__init__()
        
        # MLP that collapses state dimension and is broadcast positions
        self.position_scale = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, 1) # scalar
        )

        # MLP to scale vector features channelwise (equivariant legal scalign)
        self.vector_scale = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, s, v, pos):
        """
        s   : [N, F]       scalar node features
        v   : [N, F, 3]    vector node features
        pos : [N, 3]       atomic positions (any origin)
        """

        q = self.position_scale(s).squeeze(-1) # [N]
        g = self.vector_scale(s).unsqueeze(-1) # [N, F, 1]
        mu_atom = (v * g).sum(dim=1) # [N, 3]

        out = mu_atom + q.unsqueeze(-1) * pos  # [N, 3]
        return out