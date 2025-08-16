import torch
def predict_x0_from_eps(x_t, t, eps, alpha_bar):
    sqrt_recip_alpha_bar = torch.rsqrt(alpha_bar[t])[:, None, None, None]
    sqrt_recipm1_alpha_bar = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
    return sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * eps

