import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct


def dct_2d_numpy(x_np):
    x_np = dct(x_np, axis=-1, type=2, norm='ortho')  # width
    x_np = dct(x_np, axis=-2, type=2, norm='ortho')  # height
    return x_np


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


class AvgOptimizationTracker:
    def __init__(self, h: int, w: int, k: int = 32):
        self.k = k
        self.h = h
        self.w = w
        self.num_steps = 1e-6
        self._avg_attention_sum = torch.ones(h, w)
        self.training = True  # mimic PyTorch behavior

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @property
    def avg_attention(self):
        return self._avg_attention_sum / self.num_steps

    def forward(self, x):
        """
        If in training mode, apply random k attention samples and return all predictions (k, 2).
        Otherwise use average attention and return (2,)
        """
        # DCT once
        x_np = x.detach().cpu().numpy()
        x_dct = dct_2d_numpy(x_np)
        x_dct = torch.tensor(x_dct, device=x.device)
        gray = x_dct.mean(dim=0)  # (H, W)

        if self.training:
            # Random attention: (k, H, W)
            self._rand_attn = torch.rand(self.k, self.h, self.w, device=x.device)
            weighted = gray.unsqueeze(0) * self._rand_attn  # (k, H, W)

            row_output = weighted.mean(dim=2).mean(dim=-1)  # (k,)
            col_output = weighted.mean(dim=3).mean(dim=-1)  # (k,)

            preds = torch.stack([col_output, row_output], dim=1)  # (k, 2)
            return torch.sigmoid(preds)
        else:
            attn = self.avg_attention.to(x.device)
            weighted = gray * attn  # (H, W)

            row_output = weighted.mean(dim=1)
            col_output = weighted.mean(dim=0)

            out = torch.sigmoid(torch.stack([col_output.mean(), row_output.mean()]))  # (2,)
            return out

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def update(self, x, target):
        self.train()

        target = target.unsqueeze(-1)

        preds = self.forward(x)  # (k, 2), (k, H, W)
        errors = preds - target  # (k, 2)
        scalar_errors = errors.mean(dim=1).view(self.k, 1, 1)  # (k, 1, 1)

        weighted_attn = self._rand_attn * scalar_errors  # (k, H, W)
        avg_attention = weighted_attn.mean(dim=0)  # (H, W)

        self.num_steps += 1
        self._avg_attention_sum += avg_attention.detach()

        return scalar_errors.mean().item(), self.predict(x)


if __name__ == "__main__":
    model = AvgOptimizationTracker(h=64, w=64, k=16)
    x = torch.rand(64, 64)
    t = torch.rand(2)
    pred, err = model.update(x, t)
    print(pred, err)

    pred = model.predict(x)
    print(pred)