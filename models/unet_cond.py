import torch
import torch.nn as nn
import torch.nn.functional as F
from time_embedding import get_timestep_embedding

class BasicResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = BasicResidualBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.pool(x), x  # 输出 x 和 skip 连接

# 🔹 上采样模块
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = BasicResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class UNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()
        # 时间步编码器（可学习部分）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        # 下采样路径
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # 中间
        self.middle = BasicResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # 上采样路径
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, time_emb_dim)

        # 输出层
        self.output_conv = nn.Conv2d(base_channels, in_channels, 1)
        self.time_emb_dim = time_emb_dim

    def forward(self, x_t, t, cond):
        """
        x_t: 当前含噪图像 [B, 3, H, W]
        t: 当前时间步整数 [B]
        cond: 条件图像 [B, 3, H, W]（如 Ki67）
        """
        x = torch.cat([x_t, cond], dim=1)  # 拼接条件图像
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # 变换后嵌入 [B, time_emb_dim]

        x = self.input_conv(x)
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)

        x = self.middle(x, t_emb)

        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        return self.output_conv(x)  # 输出预测噪声 ε̂
