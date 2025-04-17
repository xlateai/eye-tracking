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


@torch.no_grad()
class _SingleFMCTracker(nn.Module):
    def __init__(self, h, w, lr: float):
        super().__init__()
        self.lr = lr
        self.attention = nn.Parameter(torch.rand(h, w))
        self.row_weights = nn.Parameter(torch.rand(h))
        self.col_weights = nn.Parameter(torch.rand(w))
        self.loss_fn = nn.L1Loss()

    @torch.no_grad()
    def forward(self, x):
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
    
    @torch.no_grad()
    def distance_to(self, other: "_SingleFMCTracker"):
        # loop through all my paramters and calculate the distance to the other's parameters
        # using distance squared
        distance = 0.0
        for param, other_param in zip(self.parameters(), other.parameters()):
            distance += torch.sum((param - other_param) ** 2)
        return distance.item()
    
    @torch.no_grad()
    def clone_to(self, other: "_SingleFMCTracker"):
        # loop through all my paramters and clone the other's parameters
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data = other_param.data.clone()
        return self
    
    @torch.no_grad()
    def perturb(self):
        # basically, just add some random noise to our paramters centered around 1
        for param in self.parameters():
            param.data = param.data + (torch.randn_like(param) * self.lr)
        return self

@torch.no_grad()
class FMCTracker(nn.Module):
    def __init__(self, h, w, k: int = 16, lr: float = 0.1, top_k: int = 3):
        """
        Initializes a population of FMC trackers.

        Args:
            h (int): Height of the input.
            w (int): Width of the input.
            k (int): Number of trackers.
            lr (float): Learning rate for perturbation.
            top_k (int): Number of top trackers to freeze from cloning and perturbation.
        """
        super().__init__()

        if top_k >= k:
            raise ValueError(f"`top_k` must be less than `k`. Got top_k={top_k}, k={k}")

        self.best_i = 0
        self.k = k
        self.top_k = top_k
        self.trackers = nn.ModuleList([_SingleFMCTracker(h, w, lr=lr) for _ in range(k)])

    @torch.no_grad()
    def calculate_distances(self):
        partners = torch.randint(0, self.k, (self.k,))
        distances = torch.zeros(self.k)
        for i in range(self.k):
            distances[i] = self.trackers[i].distance_to(self.trackers[partners[i]])
        return partners, distances

    @torch.no_grad()
    def _preproc(self, x):
        x_np = x.detach().cpu().numpy()
        x_dct = dct_2d_numpy(x_np)
        return torch.tensor(x_dct, device=x.device)

    @torch.no_grad()
    def update(self, x, target_xy):
        x = self._preproc(x)

        losses = torch.zeros(self.k)
        for i, tracker in enumerate(self.trackers):
            preds = tracker.forward(x)
            loss = tracker.loss_fn(preds.squeeze(), target_xy.squeeze())
            losses[i] = loss

        self.best_i = torch.argmin(losses)
        print("Best agent:", self.best_i.item(), "Loss:", losses[self.best_i].item())

        topk_indices = torch.topk(losses, self.top_k, largest=False).indices.tolist()

        partners, distances = self.calculate_distances()

        scores = _relativize_vector(-losses)
        distances = _relativize_vector(distances)
        vrs = scores * distances
        pair_vrs = vrs[partners]

        probability_to_clone = (pair_vrs - vrs) / torch.where(vrs > 0, vrs, 1e-8)
        r = torch.rand(self.k)
        will_clone = (r < probability_to_clone).float()

        # prevent cloning into any of the top_k agents
        for i in topk_indices:
            will_clone[i] = 0

        # execute cloning
        for i in range(self.k):
            if will_clone[i] > 0:
                self.trackers[i].clone_to(self.trackers[partners[i]])

        # perturb only if not frozen
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