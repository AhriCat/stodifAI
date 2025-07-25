import torch
import torch.nn as nn

class DriftDiffusionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu = nn.Linear(64, 1)
        self.sigma = nn.Linear(64, 1)

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        h = self.hidden(inputs)
        mu = self.mu(h)
        sigma = torch.exp(self.sigma(h))  # Ensure positivity
        return mu, sigma
