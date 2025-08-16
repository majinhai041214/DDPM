"""
正向过程（加噪声）
"""
import torch


def q_sample(x_start,t,noise,alpha_bar):
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:,None,None,None]
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
    return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise