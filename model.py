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

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=9, stride=5),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.predictor = torch.nn.Sequential(
            # adapative pooling
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 2),
        )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        emb = self.encoder(x)
        print('emb shape', emb.shape)
        preds = self.predictor(emb)
        return torch.sigmoid(preds)

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
