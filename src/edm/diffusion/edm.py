import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool
from edm.diffusion import LinearNoiseSchedule, CosineNoiseSchedule
from edm.models import PaiNNDiffusion, PaiNNConditionalDiffusion


class EDM(nn.Module):
    def __init__(
            self,
            noise_schedule,
            num_rounds=9,
            state_dim=256,
            cutoff_painn=5,
            edge_dim=64,
            T=1000
    ):
        
        super().__init__()
        self.ns = noise_schedule
        self.num_rounds = num_rounds
        self.state_dim = state_dim
        self.edge_dim = edge_dim
        self.T = T

        self.loss = nn.MSELoss()
        
        self.model = PaiNNDiffusion( 
            num_rounds,
            state_dim,
            cutoff_painn,
            edge_dim
        )


    def loss_fn(self, data):
        
        data_noised = data.clone()

        batch_size = data_noised.num_graphs if hasattr(data_noised, 'num_graphs') else 1
        t_graph = torch.randint(0, self.T + 1, size=(batch_size, ), device=data_noised.pos.device)
        
        t_atom = t_graph[data_noised.batch] # discrete [0, 1000] node level
        t_norm = t_graph.float() / self.T # [0, 1] graph level
        t_norm_atom = t_norm[data_noised.batch] # [0, 1] node level

        # Draw noise 
        eps_x = torch.randn_like(data_noised.pos)
        eps_h = torch.randn_like(data_noised.h) # Idk if this is the way we'll do it, but let's stick with it

        # Subtract CoG in a batch-aware fashion (see func below)
        eps_x = self.subtract_CoG(data_noised.batch, eps_x)

        # Encode and overwrite
        zt_x, zt_h = self.zt_given_x(data_noised.pos, data_noised.h, eps_x, eps_h, t_atom)
        data_noised.pos = zt_x
        data_noised.h   = zt_h

        # model prediction
        equi_out, inv_out = self.model(data_noised, t_norm_atom)
        
        # Parametrisation in (3.2 of EDM)
        eps_x_hat = equi_out - zt_x
        eps_x_hat = self.subtract_CoG(data_noised.batch, eps_x_hat)
        eps_h_hat = inv_out

        # We don't use that whole w(0) = -1 thing, we just ignore it.
        loss_x = self.loss(eps_x_hat, eps_x)
        loss_h = self.loss(eps_h_hat, eps_h)
        
        return loss_x + loss_h

    def zt_given_x(self, x, h, eps_x, eps_h, t):
        # Broadcast noise schedule to other dimensions for easy multiplication
        zt_x = self.ns.alphas[t][:, None] * x + self.ns.sigmas[t][:, None] * eps_x
        zt_h = self.ns.alphas[t][:, None] * h + self.ns.sigmas[t][:, None] * eps_h
        return zt_x, zt_h

    def subtract_CoG(self, batch, eps):
        means = global_mean_pool(eps, batch)
        centered = eps - means[batch]
        return centered

    def one_hot(self, h):
        return self.dataset.one_hot_converter(h)
    
    def forward(self, data, t_norm):
        # Does a pass of the model and parametrises as in EDM 3.2
        equi_out, inv_out = self.model(data, t_norm)
        eps_pred_x = equi_out - data.pos
        eps_pred_x = self.subtract_CoG(data.batch, eps_pred_x)
        eps_pred_h = inv_out
        return eps_pred_x, eps_pred_h

