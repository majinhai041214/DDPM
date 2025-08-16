"""
生成正向过程所需要的各项参数
"""

import torch
def make_beta_schedule(schedule='linear',n_times=1000,beta_start=1e-4,beta_end=0.02):
    return torch.linspace(beta_start,beta_end,n_times)

def get_diffusion_params(beta):
    alpha = 1.-beta
    alpha_bar = torch.cumprod(alpha,dim=0)
    alpha_bar_prev = torch.cat([torch.tensor([1.0]),alpha_bar[:-1]])

    return {
        'beta':beta,
        'alpha': alpha,
        'alpha_bar': alpha_bar,
        'alpha_bar_prev': alpha_bar_prev
    }