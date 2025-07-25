import torch
import torch.nn as nn
import torch.optim as optim
from model import DriftDiffusionNN
from data import generate_ou_data

def train_model(device, n_epochs=1000, lr=1e-3):
    t, x = generate_ou_data()
    t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    dt = t[1] - t[0]

    model = DriftDiffusionNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i in range(1, len(t)):
            t_input = t_tensor[i-1]
            x_prev = x_tensor[i-1]
            x_target = x_tensor[i]
            mu, sigma = model(t_input, x_prev)
            x_pred_mean = x_prev + mu * dt
            loss = loss_fn(x_pred_mean, x_target)
            total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / (len(t) - 1)}")
    return model, t_tensor, dt
