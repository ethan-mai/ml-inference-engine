"""Defines a Linear MLP Pytorch model trained on MNIST data to ~98% accuracy.

Model weights is saved as a .pt file ('mlp_model.pt') and the model metric statistics
are saved as metrics.json, with accuracy guard for a 97% accuracy threshold.
"""

import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

torch.manual_seed(42)

class MLPClassifier(nn.Module):
    # architecture: 784 -> 128 -> 64 -> 10, ReLU x2, logits out.
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def graph_layers(self):
        # retains original truth of IR node order and names as a list.
        return [
            ("fc1", self.fc1),
            ("relu1", self.relu1),
            ("fc2", self.fc2),
            ("relu2", self.relu2),
            ("fc3", self.fc3),
        ]

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    return running_loss / len(dataloader)

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


if __name__ == "__main__":
    #converts python PIL image to native pytorch tensor, before scaling
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_data = datasets.MNIST(
        root="data", train=True, download=False, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=False, transform=transform
    )

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    model = MLPClassifier().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_losses = []
    test_losses = []

    epochs = 8
    accuracy = 0.0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer, device)
        test_loss, accuracy = test(test_loader, model, loss_fn, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print("Done!")


    torch.save(model.cpu().state_dict(), "models/mlp_model.pt")
    print("Saved PyTorch model state to models/mlp_model.pt")


    metrics = {
        "arch": "784-128-64-10",
        "seed": 42,
        "epochs": epochs,
        "device": device,
        "test_top1": accuracy,
        "test_loss": test_losses[-1],
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to models/metrics.json (test_top1={accuracy:.4f})")

    if accuracy < 0.97:
        raise SystemExit(
            f"GATE FAILED: test_top1 {accuracy:.4f} < 0.97. "
            "Raise epochs; the architecture is pinned by the IR spec."
        )
