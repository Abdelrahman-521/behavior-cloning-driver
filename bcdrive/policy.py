"""Neural-network driving policies and how they're trained."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .robot import MAX_V, MAX_W, N_BEAMS

N_INPUTS = N_BEAMS + 1   # 20 lidar distances + heading error


class Policy:
    """An MLP from observation to (v, w), with input/output scaling baked in."""

    def __init__(self, hidden=(10,), normalise=False):
        layers, size = [], N_INPUTS
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU()]
            size = h
        layers.append(nn.Linear(size, 2))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        self.normalise = normalise
        self.x_mean = torch.zeros(N_INPUTS)
        self.x_std = torch.ones(N_INPUTS)
        self.y_scale = torch.tensor([MAX_V, MAX_W]) if normalise else torch.ones(2)

    def fit_scaler(self, X):
        if self.normalise:
            X = torch.as_tensor(X, dtype=torch.float32)
            self.x_mean = X.mean(0)
            self.x_std = X.std(0).clamp_min(1e-3)

    def forward(self, X):
        return self.net((X - self.x_mean) / self.x_std) * self.y_scale

    def act(self, obs):
        with torch.no_grad():
            v, w = self.forward(torch.as_tensor(obs, dtype=torch.float32)[None])[0].tolist()
        return float(np.clip(v, 0.0, MAX_V)), float(np.clip(w, -MAX_W, MAX_W))

    def state_dict(self):
        return {"net": self.net.state_dict(), "hidden": self.hidden, "normalise": self.normalise,
                "x_mean": self.x_mean, "x_std": self.x_std, "y_scale": self.y_scale}

    @classmethod
    def from_state_dict(cls, sd):
        p = cls(sd["hidden"], sd["normalise"])
        p.net.load_state_dict(sd["net"])
        p.x_mean, p.x_std, p.y_scale = sd["x_mean"], sd["x_std"], sd["y_scale"]
        return p


def train_course_style(X, Y, epochs=100, lr=0.05, seed=0):
    """The training loop from the COMP 4766 assignment (Abdel's code).

    The setup the assignment specified: one hidden layer of 10 ReLU units,
    full-batch Adam on mean-squared error, 100 epochs.
    """
    torch.manual_seed(seed)
    policy = Policy(hidden=(10,), normalise=False)
    model = policy.net
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return policy, {"train": losses}


def train(X, Y, hidden=(64, 64), epochs=300, lr=1e-3, batch_size=256, val_fraction=0.15,
          patience=30, seed=0, policy=None):
    """Improved training: scaled inputs and outputs, mini-batches, a held-out
    validation split and early stopping on validation loss."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_fraction))
    val, tr = idx[:n_val], idx[n_val:]

    if policy is None:
        policy = Policy(hidden=hidden, normalise=True)
    policy.fit_scaler(X[tr])
    Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
    Xv, Yv = torch.tensor(X[val]), torch.tensor(Y[val])
    scale = policy.y_scale

    optimizer = torch.optim.Adam(policy.net.parameters(), lr=lr)
    best, best_state, since_best = float("inf"), None, 0
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        policy.net.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for start in range(0, len(Xt), batch_size):
            b = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = (((policy.forward(Xt[b]) - Yt[b]) / scale) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        policy.net.eval()
        with torch.no_grad():
            val_loss = (((policy.forward(Xv) - Yv) / scale) ** 2).mean().item()
        history["train"].append(total / len(Xt))
        history["val"].append(val_loss)
        if val_loss < best - 1e-6:
            best, since_best = val_loss, 0
            best_state = {k: v.clone() for k, v in policy.net.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    policy.net.load_state_dict(best_state)
    return policy, history
