import torch
import torch.nn as nn
import torch.nn.init as init


class GatedEquivariantBlock(nn.Module):
    def __init__(
            self,
            state_dim=128,
        ):
        super().__init__()

        self.state_dim = state_dim

        # self.W = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
        self.W = nn.Parameter(torch.empty(self.state_dim, self.state_dim))
        init.xavier_normal_(self.W)

        self.scalar_path = nn.Sequential(
            nn.Linear(2*state_dim, 2*state_dim),
            nn.SiLU(),
            nn.Linear(2*state_dim, 2*state_dim),
        )

    def forward(self, s, v):
        """
        Following schematic in Figure 3 in PaiNN paper
        """
        Wv = self.W[None, :, :] @ v # check dimensions
        Wv_norm = torch.norm(Wv, dim=2) # check dims

        scalar_input = torch.cat((s, Wv_norm), dim=1)
        split = self.scalar_path(scalar_input)

        # Splitting into three groups
        s_delta, v_hadamard = torch.split(
            split,
            self.state_dim,
            dim=1
        )

        # print(f'Wv.shape: {Wv.shape}')
        # print(f'v_hadamard.shape: {v_hadamard.shape}')
        v_delta = Wv * v_hadamard[:, :, None]

        return s + s_delta, v + v_delta
