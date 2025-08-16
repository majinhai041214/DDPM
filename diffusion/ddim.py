"""
DDIM只用于推理过程
"""


import torch
from .reverse_sample import predict_x0_from_eps, ddim_step

@torch.no_grad()
def ddim_sample(model, cond, alpha_bar, shape, device, eta=0.0, num_steps=50):
    """
    使用 DDIM 从 x_T 逐步反向采样，最终得到 x_0

    参数：
        model: 噪声预测模型
        cond: 条件图像（如 Ki67），shape = [B, 3, H, W]
        alpha_bar: alpha_bar 序列，形状为 [T]
        shape: 要生成图像的形状，如 (B, 3, 256, 256)
        device: torch.device("cuda") or "cpu"
        eta: 控制采样的随机性，0 表示确定性
        num_steps: 推理时的采样步数（推荐 50）

    返回：
        最终还原出的 x_0 图像张量
    """
    T = len(alpha_bar)
    step_size = T // num_steps
    time_steps = list(range(0, T, step_size))[-num_steps:]
    time_steps.reverse()  # 从 T 开始往回走

    x_t = torch.randn(shape, device=device)

    for t in time_steps:
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x_t, t_tensor, cond)  # 模型预测噪声
        x_t = ddim_step(x_t, t_tensor, eps, alpha_bar, eta=eta)  # 得到 x_{t-1}

    return x_t  # 最终图像（即 x_0）

