import torch
import torch.nn as nn
import math


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embeddings for timesteps
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ContextEmbedding(nn.Module):
    """
    Time embedding for diffusion model
    """
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.sinusoidal = SinusoidalEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))