import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct


def dct_2d_numpy(x_np):
    x_np = dct(x_np, axis=-1, type=2, norm='ortho')  # width
    x_np = dct(x_np, axis=-2, type=2, norm='ortho')  # height
    return x_np


def attention_to_image(attention: torch.Tensor) -> np.ndarray:
    att = attention.detach().cpu().numpy()
    att = (att - att.min()) / (att.ptp() + 1e-6)
    att_img = (att * 255).astype(np.uint8)
    rgba = np.stack([att_img] * 3 + [np.full_like(att_img, 255)], axis=-1)
    return rgba


class EfficientEyeTracker(nn.Module):
    def __init__(self, h, w, lr=0.1):
        super().__init__()
        self.attention = nn.Parameter(torch.ones(h, w))
        self.row_weights = nn.Parameter(torch.ones(h))
        self.col_weights = nn.Parameter(torch.ones(w))
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        x_dct = dct_2d_numpy(x_np)
        x = torch.tensor(x_dct, device=x.device)

        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        weighted = gray.squeeze(1) * self.attention  # (B, H, W)

        row_sum = weighted.mean(dim=2)  # (B, H)
        col_sum = weighted.mean(dim=1)  # (B, W)

        row_output = (row_sum * self.row_weights).mean(dim=1)  # (B,)
        col_output = (col_sum * self.col_weights).mean(dim=1)  # (B,)

        return torch.sigmoid(torch.stack([col_output, row_output], dim=1))  # (B, 2)

    def update(self, x, target_xy):
        pred = self(x)
        loss = self.loss_fn(pred.squeeze(), target_xy.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred.detach()
