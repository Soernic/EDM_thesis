import torch
import torch.nn as nn

from pdb import set_trace


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
    

class AlphaEquivariantReadout(nn.Module):
    """
    Following the alpha specific regressor head in PaiNN
    """
    def __init__(self, state_dim: int):
        super().__init__()

        self.alpha_0 = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, 1) # scalar            
        )

        self.nu = VectorLinear(state_dim)

    def forward(self, s, v, pos):        
        alpha_0 = self.alpha_0(s) # [B, 1]
        nu = self.nu(v) # [B, 3] (feature dimension is collapsed in VectorLinear)

        B = nu.shape[0]
        I = torch.eye(3, device=nu.device)[None, :, :].expand(B, 3, 3) # [B, 3, 3]

        # Outer products using einsum
        outer1 = torch.einsum('bi,bj->bij', nu, pos)  # [B, 3, 3]
        outer2 = torch.einsum('bi,bj->bij', pos, nu)  # [B, 3, 3]

        return alpha_0[:, :, None] * I + outer1 + outer2

        
class VectorLinear(nn.Module):
    """
    A way to collapse dimensions in a vector feature of shape [batch, state_dim, spatial_dim]
    """
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.weights = nn.Parameter(torch.ones(state_dim))

    def forward(self, v):
        v_weighted = v * self.weights[None, :, None]
        out = v_weighted.sum(dim=1)
        return out




