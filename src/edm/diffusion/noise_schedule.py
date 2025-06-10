import torch


class CosineNoiseSchedule:
    """
    Cosine schedule as defined in appendix B of EDM. 
    Noise schedule definition for alpha_t, sigma_t for t in {0, ..., T}
    alpha(t) = sqrt(1 - sigma(t)^2)

    Let
    alpha(t) := (1 - 2s)f(t) + s
    f(t)     := 1 - (t/T)^2
    and default s := 1e-5. 

    Additional tricks: 
    - Let alpha_{t|s} := alpha(t) / alpha(s)
    - Let alpha_{-1} := 1
    - Let alpha_{t|s}^2 := clamp(alpha_{t|s}, min=0.001) to prevent numerical instability
    - Let sigma_{t|s}^2 := sigma(t)^2 - alpha_{t|s}^2 * sigma(s)^2
    - Let sigma_{t \to s} := sigma_{t|s} sigma(s) / sigma(t)
    - Let SNR(t) := alpha(t)^2 / sigma(t)^2

    We have to be a bit careful with cosine schedule for stability reasons. 
    There's a lot of clipping going on, since it explodes at the edges otherwise
    """
    def __init__(self, T=1000, s=1e-5, device='cpu'):
        self.T = T
        self.s = s
        self.device = device

        t = torch.arange(0, T+1, device=device)

        # Raw alpha, i.e., now yet clipped as stated in the paper
        alpha_raw = ((1.0 - 2.0 * s) * (1.0 - (t/T)**2) + s).clamp(0.0, 1.0)

        # Step-wise clipping
        # Essentially, compute all the ratios alpha(t)^2 / alpha(s)^2, then clip them 
        # and back-calculate everything else
        alpha2_raw = alpha_raw**2 # alpha(t)^2
        step_ratio = alpha2_raw[1:] / alpha2_raw[:-1] # r(t)
        step_ratio = torch.clamp(step_ratio, min=0.001, max=1.0) # clip it
        alpha2 = torch.cumprod(torch.cat([torch.ones(1, device=device), step_ratio], dim=0), dim=0)

        self.alphas = torch.sqrt(alpha2)
        self.sigmas = torch.sqrt(1.0 - self.alphas**2)


    def alpha(self, t):
        # We've defined alpha(-1) := 1
        out = self.alphas[t.clamp(min=0)] # make sure we don't break things
        return torch.where(t==-1, torch.ones_like(out), out)

    def sigma(self, t):
        return self.sigmas[t.clamp(min=0)] # not sure this is necessary, but just to be safe

    # Consecutive-step ratio (s=t-1), always within [sqrt(0.001), 1]
    def alpha_t_given_s(self, t, s):
        return self.alpha(t) / self.alpha(s)

    def sigma_t_given_s(self, t, s):
        return torch.sqrt(self.sigma(t)**2 - self.alpha_t_given_s(t, s)**2 * self.sigma(s)**2)

    def sigma_t_to_s(self, t, s):
        return self.sigma_t_given_s(t, s)*self.sigma(s) / self.sigma(t)

    def SNR(self, t):
        return self.alpha(t)**2 / self.sigma(t)**2
    
    def c1(self, t, s):
        1 / self.alpha_t_given_s(t, s)

    def c2(self, t, s):
        alpha_ts = self.alpha_t_given_s(t, s)
        sigma_ts = self.sigma_t_given_s(t, s)
        return -(sigma_ts**2) / (alpha_ts * self.sigma(t))
    
    def c3(self, t, s):
        return self.sigma_t_to_s(t, s)



class LinearNoiseSchedule:
    """
    Classic linear DDPM-like schedule
    Modification: For DDPM, the noise schedule has 
    - beta
    - alpha
    - alpha_bar

    .. too many variables. We just use 
    - beta: same as DDPM
    - alpha: sqrt(alpha_bar)
    
    This means the forward process can be described as 
    q(xt|x0) = N(xt; alpha(t)x0, (1 - alpha(t)**2)I)
    This is consistent with EDM notation for the cosine-like schedule.
    """
    
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.02, device='cpu'):
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        
        betas_line = torch.linspace(beta_min, beta_max, T, device=device)
        self.betas = torch.cat([torch.zeros(1, device=device), betas_line], dim=0)

        alpha_bars = torch.cumprod(1.0 - self.betas, dim=0)
        self.alphas = torch.sqrt(alpha_bars)
        self.sigmas = torch.sqrt(1.0 - alpha_bars)

    def beta(self, t):
        return self.betas[t]

    def alpha(self, t):
        return self.alphas[t]

    def sigma(self, t):
        return torch.sqrt(1 - self.alpha(t)**2)

    def alpha_t_given_s(self, t, s):
        return self.alpha(t) / self.alpha(s)

    def sigma_t_given_s(self, t, s):
        return torch.sqrt(self.sigma(t)**2 - self.alpha_t_given_s(t, s)**2 * self.sigma(s)**2)

    def SNR(self, t):
        return self.alpha(t)**2 / self.sigma(t)**2

    def sigma_t_to_s(self, t, s):
        return self.sigma_t_given_s(t, s)*self.sigma(s) / self.sigma(t)

    def c1(self, t, s):
        1 / self.alpha_t_given_s(t, s)

    def c2(self, t, s):
        alpha_ts = self.alpha_t_given_s(t, s)
        sigma_ts = self.sigma_t_given_s(t, s)
        return -(sigma_ts**2) / (alpha_ts * self.sigma(t))
    
    def c3(self, t, s):
        return self.sigma_t_to_s(t, s)    