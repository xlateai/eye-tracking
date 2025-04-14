import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.num_steps = 1
        self.sum_fc1 = torch.zeros_like(self.fc1.weight)
        self.sum_fc2 = torch.zeros_like(self.fc2.weight)

    def forward(self, x, w1=None, w2=None):
        x = x.view(-1, 28*28)
        h = F.relu(x @ (self.fc1.weight.T if w1 is None else w1.T))
        return h @ (self.fc2.weight.T if w2 is None else w2.T)

    def update(self, x, y, k=32):
        x = x.view(-1, 28*28)
        w1s = [torch.randn_like(self.fc1.weight) for _ in range(k)]
        w2s = [torch.randn_like(self.fc2.weight) for _ in range(k)]

        losses = []
        for i in range(k):
            logits = self.forward(x, w1s[i], w2s[i])
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())

        # Convert to probabilities: lower loss = higher weight
        losses = torch.tensor(losses)
        probs = torch.softmax(-losses, dim=0)  # or: 1/loss and normalize manually

        # Compute weighted average model
        w1_avg = sum(p * w for p, w in zip(probs, w1s))
        w2_avg = sum(p * w for p, w in zip(probs, w2s))

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
