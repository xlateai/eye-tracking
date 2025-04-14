import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32, bias=False)
        self.fc2 = nn.Linear(32, 10, bias=False)
        self.num_steps = 1
        self.sum_fc1 = torch.zeros_like(self.fc1.weight)
        self.sum_fc2 = torch.zeros_like(self.fc2.weight)

    def forward(self, x, w1=None, w2=None):
        x = x.view(-1, 28*28)
        h = F.relu(x @ (self.fc1.weight.T if w1 is None else w1.T))
        return h @ (self.fc2.weight.T if w2 is None else w2.T)

    def update(self, x, y, k=16):
        x = x.view(-1, 28*28)  # (B, 784)
        B = x.size(0)

        w1s = torch.randn(k, *self.fc1.weight.shape, device=x.device)  # (k, 128, 784)
        w2s = torch.randn(k, *self.fc2.weight.shape, device=x.device)  # (k, 10, 128)

        # Forward pass over k sampled models
        h = F.relu(torch.einsum('b i, k o i -> k b o', x, w1s))        # (k, B, 128)
        logits = torch.einsum('k b i, k o i -> k b o', h, w2s)         # (k, B, 10)

        # Reshape logits: (k * B, 10), target: (k * B,)
        logits = logits.view(-1, 10)                                   # (k*B, 10)
        targets = y.repeat(k)                                          # (k*B,)

        # Compute loss
        losses = F.cross_entropy(logits, targets, reduction='none')   # (k*B,)
        losses = losses.view(k, B).mean(dim=1)                         # (k,)

        # Convert to softmax weights: lower loss = higher weight
        probs = torch.softmax(-losses, dim=0)                          # (k,)

        # Weighted average of sampled weights
        w1_avg = torch.einsum('k, k o i -> o i', probs, w1s)           # (128, 784)
        w2_avg = torch.einsum('k, k o i -> o i', probs, w2s)           # (10, 128)

        self.sum_fc1 += w1_avg
        self.sum_fc2 += w2_avg
        self.num_steps += 1

        self.fc1.weight.data = self.sum_fc1 / self.num_steps
        self.fc2.weight.data = self.sum_fc2 / self.num_steps

# Load data
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, download=True, transform=transform), batch_size=1000)

# Train loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
for epoch in range(1):
    for x, y in train_loader:
        model.update(x.to(device), y.to(device))

# Evaluate
model.eval(); correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
print(f"Test accuracy: {correct / len(test_loader.dataset):.4f}")
