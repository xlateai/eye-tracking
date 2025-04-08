import torch
import torch.nn as nn


class EfficientEyeTracker(nn.Module):
    def __init__(self, h, w, lr=0.2):
        """
        Initialize the EfficientEyeTracker model and its training components.

        Args:
            h (int): Height of the input image
            w (int): Width of the input image
            lr (float): Learning rate for the optimizer
        """
        super().__init__()
        self.attention = nn.Parameter(torch.ones(h, w))
        self.row_weights = nn.Parameter(torch.ones(h))
        self.col_weights = nn.Parameter(torch.ones(w))

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        weighted = x * self.attention  # (1, H, W)
        row_sum = weighted.mean(dim=2)  # (1, H)
        col_sum = weighted.mean(dim=1)  # (1, W)
        row_output = (row_sum * self.row_weights).mean(dim=1)  # (1,)
        col_output = (col_sum * self.col_weights).mean(dim=1)  # (1,)
        output = torch.stack([col_output, row_output], dim=1)  # (1, 2)
        return torch.sigmoid(output)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def update(self, x, target):
        """
        Perform one training step.

        Args:
            x (Tensor): Input tensor of shape (1, H, W)
            target (Tensor): Target tensor of shape (1, 2)

        Returns:
            float: Loss value, Tensor: prediction
        """
        self.train()
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item(), pred
