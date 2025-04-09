import torch
import torch.nn as nn



def attn_forward(x, attention, row_weights, col_weights):
    weighted = x * attention  # (1, H, W)
    row_sum = weighted.mean(dim=2)  # (1, H)
    col_sum = weighted.mean(dim=1)  # (1, W)
    row_output = (row_sum * row_weights).mean(dim=1)  # (1,)
    col_output = (col_sum * col_weights).mean(dim=1)  # (1,)
    output = torch.stack([col_output, row_output], dim=1)  # (1, 2)
    return torch.sigmoid(output)


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
        return attn_forward(x, self.attention, self.row_weights, self.col_weights)
    
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


class AvgOptimizationTracker:
    def __init__(self, h: int, w: int, k: int):
        """
        This model works by randomizing the weights at each train prediction.
        The randomized weights are then weighted by their performance and tracked using a running
        average to determine the best weights.
        """

        self.k = k
        self.h = h
        self.w = w
        self.num_steps = 0
        self._avg_attention_sum = torch.ones(h, w)
        self._avg_row_weights_sum = torch.ones(h)
        self._avg_col_weights_sum = torch.ones(w)

    @property
    def avg_attention(self):
        return self._avg_attention_sum / self.num_steps
    
    @property
    def avg_row_weights(self):
        return self._avg_row_weights_sum / self.num_steps
    
    @property
    def avg_col_weights(self):
        return self._avg_col_weights_sum / self.num_steps
    
    def update(self, x, target):
        """
        Perform one training step.

        Args:
            x (Tensor): Input tensor of shape (1, H, W)
            target (Tensor): Target tensor of shape (1, 2)

        Returns:
            float: Loss value, Tensor: prediction
        """

        # random uniform weights k, h, w for atn
        rand_attention = torch.rand(self.k, self.h, self.w)
        rand_row_weights = torch.rand(self.k, self.h)
        rand_col_weights = torch.rand(self.k, self.w)

        preds = attn_forward(x, rand_attention, rand_row_weights, rand_col_weights)
        
        # calculate error between preds and singular target
        # NOTE: it's very important that this distance remain signed (not MAE or MSE)
        # this is to ensure that averages will result in 0 if the mean is correct
        errors = (preds - target)

        # scalar value per pred so we can form our quality weights
        errors = errors.mean(dim=1).unsqueeze(-1)

        # now, we need to find the weighted average of the attention, row_weights, and col_weights
        # with respect to these errors such that we can use them to update our running average
        # sums
        
        # basically, we need to reduce the self.k dimension to 1 by using the errors as dot-product weights
        avg_attention = torch.mean(rand_attention * errors.unsqueeze(-1), dim=0)
        avg_row_weights = torch.mean(rand_row_weights * errors, dim=0)
        avg_col_weights = torch.mean(rand_col_weights * errors, dim=0)

        print("avg_attention", avg_attention.shape)
        print("avg_row_weights", avg_row_weights.shape)
        print("avg_col_weights", avg_col_weights.shape)

        # an interesting hypothesis:
        # if we were to use these average weights to calculate predictions,
        # the error should approximately be equal to the average of the errors?
        # TODO: LET'S TEST THIS HYPTOTHESIS SCIENTIFICALLY!
        # would also be fun to do some mathematical analysis of this.

        # update the running average sums
        self.num_steps += 1
        self._avg_attention_sum += avg_attention
        self._avg_row_weights_sum += avg_row_weights
        self._avg_col_weights_sum += avg_col_weights

        return errors.mean(dim=0), preds.mean(dim=0)

    def predict(self, x):
        """
        Perform one prediction step.

        Args:
            x (Tensor): Input tensor of shape (1, H, W)

        Returns:
            Tensor: prediction
        """

        return attn_forward(x, self.avg_attention.unsqueeze(0), self.avg_row_weights.unsqueeze(0), self.avg_col_weights.unsqueeze(0))

    def forward(self, *args):
        raise NotImplementedError


if __name__ == "__main__":
    model = AvgOptimizationTracker(h=64, w=64, k=16)
    x = torch.rand(64, 64)
    t = torch.rand(2)
    pred, err = model.update(x, t)
    print(pred, err)

    pred = model.predict(x)
    print(pred)