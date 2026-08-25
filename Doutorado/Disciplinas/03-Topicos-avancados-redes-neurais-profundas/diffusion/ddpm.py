import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Agendador de ruído (DDPM linear)
# ----------------------------------------------------------------------------
class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        """Aplica q(x_t | x_0): amostra imagem ruidosa no passo t."""
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise


# ----------------------------------------------------------------------------
# Embedding de tempo (senoidal, como em Transformers)
# ----------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        half = self.dim // 2
        log_base = torch.log(torch.tensor(10000.0, device=t.device))
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (log_base / half)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

    
