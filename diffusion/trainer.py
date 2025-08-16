from schedule import make_beta_schedule,get_diffusion_params
import torch
import torch.nn.functional as F
from q_sampler import q_sample

class DiffusionTrainer:
    def __init__(self, model, optimizer, device,T=1000):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.T = T

        # 初始化beta序列
        beta = make_beta_schedule(schedule="linear",n_times=T)
        self.diff_params = get_diffusion_params(beta)
        self.alpha_bar = self.diff_params['alpha_bar'].to(device)

    def train_step(self,batch):
        """
        对单个 batch 进行一次训练步骤（不含结构损失）
        :return:loss.item
        """
        self.model.train()

        x0 = batch['he_data'].to(self.device)
        cond = batch['ki67_data'].to(self.device)

        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device).long()
        noise = torch.randn_like(x0)  # ε：随机噪声

        # 正向加噪，生成 xₜ
        x_t = q_sample(x_start=x0, t=t, noise=noise, alpha_bar=self.alpha_bar)
        # 模型预测 ε̂
        eps_pred = self.model(x_t, t, cond)
        # 损失 = MSE(ε̂, ε)
        loss = F.mse_loss(eps_pred, noise)

        # 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
