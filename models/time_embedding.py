import torch
import torch.nn.functional as F
import math
"""
对时间信息进行编码，便于模型感知时间信息
"""

def get_timestep_embedding(self,time_steps,embedding_dim):
    """
       将时间步 t（整数）编码为 [B, embedding_dim] 的向量。
       time_steps: LongTensor [B]  (每个样本一个时间步)
       embedding_dim: int，输出维度，通常为 128 或 256
    """
    half_dim = embedding_dim // 2
    exponents = torch.arange(half_dim, dtype=torch.float32, device=time_steps.device)
    exponents = 10000 ** (-exponents / half_dim)

    angles = time_steps[:, None].float() * exponents[None, :]  # [B, half_dim]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim]

    if embedding_dim % 2 == 1:  # 奇数维度补1维
        emb = F.pad(emb, (0, 1))

    return emb  # [B, embedding_dim]


