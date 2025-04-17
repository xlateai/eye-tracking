import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct
from collections import deque


def dct_2d_numpy(x_np):
    x_np = dct(x_np, axis=-1, type=2, norm='ortho')  # width
    x_np = dct(x_np, axis=-2, type=2, norm='ortho')  # height
    return x_np


# def _relativize_vector_np(vector: np.ndarray):
#     std = vector.std()
#     if std == 0:
#         return np.ones(len(vector))
#     standard = (vector - vector.mean()) / std
#     standard[standard > 0] = np.log(1 + standard[standard > 0]) + 1
#     standard[standard <= 0] = np.exp(standard[standard <= 0])
#     return standard

def _relativize_vector(vector: torch.Tensor):
    std = vector.std()
    if std == 0:
        return torch.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = torch.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard


@torch.no_grad()
class _SingleFMCTracker(nn.Module):
    def __init__(self, h, w, lr: float, time_horizon: int = 10):
        super().__init__()
        self.lr = lr
        self.attention = nn.Parameter(torch.rand(h, w))
        self.row_weights = nn.Parameter(torch.rand(h))
        self.col_weights = nn.Parameter(torch.rand(w))
        self._loss_fn = nn.L1Loss()
        self.loss_history = deque([0.0] * time_horizon, maxlen=time_horizon)
        self.time_horizon = time_horizon

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        weighted = gray.squeeze(1) * self.attention  # (B, H, W)

        row_sum = weighted.mean(dim=2)  # (B, H)
        col_sum = weighted.mean(dim=1)  # (B, W)

        row_output = (row_sum * self.row_weights).mean(dim=1)  # (B,)
        col_output = (col_sum * self.col_weights).mean(dim=1)  # (B,)

        return torch.sigmoid(torch.stack([col_output, row_output], dim=1))  # (B, 2)

    def update_loss(self, x, target_xy):
        preds = self.forward(x)
        loss = self._loss_fn(preds.squeeze(), target_xy.squeeze())
        self.loss_history.append(loss.item())
        return loss.item()

    def smoothed_loss(self):
        return sum(self.loss_history) / len(self.loss_history)

    def distance_to(self, other: "_SingleFMCTracker"):
        distance = 0.0
        for param, other_param in zip(self.parameters(), other.parameters()):
            distance += torch.sum((param - other_param) ** 2)
        return distance.item()

    def clone_to(self, other: "_SingleFMCTracker"):
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data = other_param.data.clone()
        self.loss_history = deque(other.loss_history, maxlen=self.time_horizon)
        return self

    def perturb(self):
        for param in self.parameters():
            param.data = param.data + (torch.randn_like(param) * self.lr)
        return self

@torch.no_grad()
class FMCTracker(nn.Module):
    def __init__(self, h, w, k: int = 16, lr: float = 0.1, top_k: int = 1, time_horizon: int = 10):
        if top_k >= k:
            raise ValueError(f"`top_k` must be less than `k`. Got top_k={top_k}, k={k}")
        super().__init__()
        self.best_i = 0
        self.k = k
        self.top_k = top_k
        self.trackers = nn.ModuleList([
            _SingleFMCTracker(h, w, lr=lr, time_horizon=time_horizon) for _ in range(k)
        ])

    def calculate_distances(self):
        partners = torch.randint(0, self.k, (self.k,))
        distances = torch.zeros(self.k)
        for i in range(self.k):
            distances[i] = self.trackers[i].distance_to(self.trackers[partners[i]])
        return partners, distances

    def _preproc(self, x):
        x_np = x.detach().cpu().numpy()
        x_dct = dct_2d_numpy(x_np)
        return torch.tensor(x_dct, device=x.device)

    def update(self, x, target_xy):
        x = self._preproc(x)

        losses = torch.zeros(self.k)
        for i, tracker in enumerate(self.trackers):
            tracker.update_loss(x, target_xy)
            losses[i] = tracker.smoothed_loss()

        self.best_i = torch.argmin(losses)
        print("Best agent:", self.best_i.item(), "Smoothed Loss:", losses[self.best_i].item())

        topk_indices = torch.topk(losses, self.top_k, largest=False).indices.tolist()
        partners, distances = self.calculate_distances()

        scores = _relativize_vector(-losses)
        distances = _relativize_vector(distances)
        vrs = scores * distances
        pair_vrs = vrs[partners]

        probability_to_clone = (pair_vrs - vrs) / torch.where(vrs > 0, vrs, 1e-8)
        r = torch.rand(self.k)
        will_clone = (r < probability_to_clone).float()

        for i in topk_indices:
            will_clone[i] = 0

        for i in range(self.k):
            if will_clone[i] > 0:
                self.trackers[i].clone_to(self.trackers[partners[i]])

        for i in range(self.k):
            if i not in topk_indices and will_clone[i] == 0:
                self.trackers[i].perturb()

        return losses[self.best_i].item()
    
    def forward(self, x):
        x = self._preproc(x)
        return self.trackers[self.best_i](x)

    
if __name__ == "__main__":
    # Example usage
    h, w = 64, 64
    k = 16
    model = FMCTracker(h, w, k, lr=1.0)
    
    # Calculate distances between trackers
    # distances = model.calculate_distances()
    # print("Distances:", distances)

    # random input and target data (k)
    x = torch.rand(1, 1, h, w)
    target_xy = torch.rand(2)
    # print(x, target_xy)

    for _ in range(100):
        model.update(x, target_xy)

    print(model(x), target_xy)