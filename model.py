import torch
import torch.nn as nn


class EfficientEyeTracker(nn.Module):
    def __init__(self, h, w, lr=0.1):
        """
        Initialize the EfficientEyeTracker model and its training components.

        Args:
            h (int): Height of the input image
            w (int): Width of the input image
            lr (float): Learning rate for the optimizer
        """
        super().__init__()

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=7, stride=3),
            nn.Sigmoid(),
            nn.Conv2d(3, 3, kernel_size=5, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(3, 1, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )

        # let's get the output shape of the encoder
        x = torch.randn(1, 1, h, w)
        x = self.encoder(x)
        features = x.shape.numel()
        print(features, "conv features")

        self.predictor = torch.nn.Sequential(
            # adapative pooling
            #nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(features, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
