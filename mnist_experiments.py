"""
MNIST classifier with controlled experiments (PyTorch).

Goal: change ONE thing at a time (optimizer, learning rate) with fixed seeds,
and record test accuracy so the results are reproducible and comparable.

Setup:   pip install torch torchvision pandas
Run:     python mnist_experiments.py
Output:  results.csv  (one row per experiment), printed summary table
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test, batch_size=1000))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(name)


def run_experiment(optimizer_name, lr, seed):
    set_seed(seed)
    train_loader, test_loader = get_loaders()
    model = SmallCNN().to(DEVICE)
    opt = make_optimizer(optimizer_name, model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    rows = []
    for optimizer_name, lr in [("sgd", 0.01), ("sgd", 0.1), ("adam", 0.001), ("adam", 0.01)]:
        for seed in (0, 1, 2):  # 3 seeds -> report mean and spread, not one lucky run
            acc = run_experiment(optimizer_name, lr, seed)
            rows.append({"optimizer": optimizer_name, "lr": lr, "seed": seed, "test_acc": acc})
            print(f"{optimizer_name} lr={lr} seed={seed} -> {acc:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    summary = df.groupby(["optimizer", "lr"])["test_acc"].agg(["mean", "std"]).round(4)
    print("\nSummary (mean and std over 3 seeds):")
    print(summary)
