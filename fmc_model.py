import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct


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


class _SingleFMCTracker(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.attention = nn.Parameter(torch.rand(h, w))
        self.row_weights = nn.Parameter(torch.rand(h))
        self.col_weights = nn.Parameter(torch.rand(w))
        self.loss_fn = nn.L1Loss()

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

    # def update(self, x, target_xy):
    #     pred = self(x)
    #     loss = self.loss_fn(pred.squeeze(), target_xy.squeeze())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item(), pred.detach()
    
    def distance_to(self, other: "_SingleFMCTracker"):
        # loop through all my paramters and calculate the distance to the other's parameters
        # using distance squared
        distance = 0.0
        for param, other_param in zip(self.parameters(), other.parameters()):
            distance += torch.sum((param - other_param) ** 2)
        return distance.item()

class FMCTracker(nn.Module):
    def __init__(self, h, w, k: int):
        """Initializes a population of FMC trackers.
        """

        super().__init__()

        self.k = k
        self.trackers = nn.ModuleList([_SingleFMCTracker(h, w) for _ in range(k)])

    def calculate_distances(self):
        # select random partners for each tracker
        partners = torch.randint(0, self.k, (self.k,))
        print(partners)
        distances = torch.zeros(self.k)
        for i in range(self.k):
            distances[i] = self.trackers[i].distance_to(self.trackers[partners[i]])
        return partners, distances
    
    def update(self, x, target_xy):
        # forward each agent and get their losses
        losses = torch.zeros(self.k)
        for i, tracker in enumerate(self.trackers):
            preds = tracker.forward(x)
            loss = tracker.loss_fn(preds.squeeze(), target_xy.squeeze())
            losses[i] = loss

        partners, distances = self.calculate_distances()

        # calculate the virtual rewards
        scores = _relativize_vector(-losses)
        distances = _relativize_vector(distances)
        vrs = scores * distances
        pair_vrs = vrs[partners]

        probability_to_clone = (pair_vrs - vrs) / torch.where(vrs > 0, vrs, 1e-8)
        r = torch.rand(self.k)
        will_clone = (r < probability_to_clone).float()
        print(probability_to_clone, will_clone, r)

        # get the probability to clone

        return losses
    
if __name__ == "__main__":
    # Example usage
    h, w = 64, 64
    k = 5
    model = FMCTracker(h, w, k)
    
    # Calculate distances between trackers
    distances = model.calculate_distances()
    print("Distances:", distances)

    # random input and target data (k)
    x = torch.rand(1, 1, h, w)
    target_xy = torch.rand(2)
    print(x, target_xy)

    model.update(x, target_xy)
