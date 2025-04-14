import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define model with randomized attention updates
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.num_steps = 1
        self.sum_fc1 = torch.zeros_like(self.fc1.weight)
        self.sum_fc2 = torch.zeros_like(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(x @ self.fc1.weight.T)
        return x @ self.fc2.weight.T

    def update(self, x, y, k=32):
        x = x.view(-1, 28*28)
        for _ in range(k):
            w1 = torch.randn_like(self.fc1.weight)
            w2 = torch.randn_like(self.fc2.weight)
            h = F.relu(x @ w1.T)
            logits = h @ w2.T
            loss = F.cross_entropy(logits, y)
            weight = loss.item()
            self.sum_fc1 += w1 * weight
            self.sum_fc2 += w2 * weight
            self.num_steps += 1
        self.fc1.weight.data = self.sum_fc1 / self.num_steps
        self.fc2.weight.data = self.sum_fc2 / self.num_steps

# Load data
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, download=True, transform=transform), batch_size=1000)

# Train loop (no optimizer)
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
