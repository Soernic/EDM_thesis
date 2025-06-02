import torch
import torch.nn as nn
import torch.nn.init as init



class PaiNNUpdate(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        # # TODO: Compare linear and parameter implementations
        # self.U = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
        # self.V = nn.Parameter(torch.randn(self.state_dim, self.state_dim))

        # Instead of a random normal, create empty parameters and use xavier init
        self.U = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        self.V = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        init.xavier_normal_(self.U)  # or init.xavier_normal_(self.U)
        init.xavier_normal_(self.V)  # or init.xavier_normal_(self.V)



        self.a = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim*3) # tripple it here
        )
        

    def forward(self, state, state_vec):
        
        # U-dot and V-dot
        # TODO: Sanity check this
        udot = self.U[None, :, :] @ state_vec
        vdot = self.V[None, :, :] @ state_vec

        # Norm passing to sj stack
        vdot_norm = torch.norm(vdot, dim=2)
        stack = torch.cat([state, vdot_norm], dim=1)

        # sj pass
        split = self.a(stack)


        # Splitting into three groups
        a_vv, a_sv, a_ss = torch.split(
            split,
            self.state_dim,
            dim=1
        )


        # Delta vi line
        delta_vi = udot * a_vv[:, :, None]

        # Delta si line
        dot = torch.sum(udot*vdot, dim=2)
        dot = dot * a_sv
        delta_si = dot + a_ss

        state = state + delta_si
        state_vec = state_vec + delta_vi

        return state, state_vec
