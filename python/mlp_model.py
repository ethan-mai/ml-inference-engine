import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

torch.manual_seed(42)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('mps'), y.to('mps')

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

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('mps'), y.to('mps')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


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

    model = MLPClassifier().to('mps')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_losses = []
    test_losses = []

    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer)
        test_loss = test(test_loader, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print("Done!")

    torch.save(model.state_dict(), "models/mlp_model.pt")
    print("Saved PyTorch model state to models/mlp_model.pt")