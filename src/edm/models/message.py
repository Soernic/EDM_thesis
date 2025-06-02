import torch
import torch.nn as nn
      

class PaiNNMessage(nn.Module):
    def __init__(self, state_dim, edge_dim, cutoff):
        super().__init__()
        self.state_dim = state_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff

        self.phi = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim * 3) # triple the outputs fro splitting
        )

        self.W = nn.Linear(edge_dim, state_dim*3)


    def forward(self, state, state_vec, edge, r_ij, norm_r_ij):
        # For stability, we clamp the norm to be within reasonable values (selected based on gut feeling and empirical evidence)
        norm_r_ij = norm_r_ij.clamp(min=1e-2, max=20)

        # W pass
        RBF = self.RBF(norm_r_ij)
        W = self.W(RBF)
        W = W * self.cosine_cutoff(norm_r_ij)[:, :, None]

        # phi pass
        phi = self.phi(state)

        # Combination with hadamard
        combination = phi[edge[1, :]] * W.squeeze(1)

        # Splitting into the 3 parts
        gate_state_vec, gate_edge_vec, scalar_message = torch.split(
            combination,
            self.state_dim,
            dim=1
        )

        #  Vector message part
        vector_message = state_vec[edge[1, :]] * gate_state_vec[:, :, None]
        normalised_r_ij = r_ij / (norm_r_ij+1)
        
        edge_vec = normalised_r_ij[:, None, :] * gate_edge_vec[:, :, None]
        vector_message += edge_vec # edge_vec is edge and vector interaction in the diagram

        # Sum of incoming messages
        delta_si = torch.zeros_like(state)
        delta_vi = torch.zeros_like(state_vec)
        delta_si = torch.index_add(delta_si, 0, edge[0, :], scalar_message)
        delta_vi = torch.index_add(delta_vi, 0, edge[0, :], vector_message)

        state = state + delta_si
        state_vec = state_vec + delta_vi

        return state, state_vec


    def RBF(self, norm_r_ij):
        # NOTE: see note, added +1 here for normalisation. Don't forget that in report if you use it!
        n = torch.arange(self.edge_dim, device=norm_r_ij.device) + 1
        return torch.sin(norm_r_ij.unsqueeze(-1) * n * torch.pi / self.cutoff) / (norm_r_ij.unsqueeze(-1) + 1) # NOTE: ADDED +1 for normaalisation here!!!! as a test..
    

    def cosine_cutoff(self, norm_r_ij):
        return torch.where(
            norm_r_ij < self.cutoff,
            0.5 * (torch.cos(torch.pi * norm_r_ij / self.cutoff) + 1),
            torch.tensor(0.0, device=norm_r_ij.device, dtype=norm_r_ij.dtype),            
        )
