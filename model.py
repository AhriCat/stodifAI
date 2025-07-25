import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate sample data from an Ornstein-Uhlenbeck process
def generate_ou_data(theta=0.7, mu=0.0, sigma=0.3, x0=1.0, T=10.0, N=1000):
    dt = T / N
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return t, x

t, x = generate_ou_data()
t_tensor = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)

# Neural network to model the OU process
class DriftDiffusionNN(nn.Module):
    def __init__(self):  # Fixed: proper __init__ method
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 64),  # Fixed: input size is 2 (time and state)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu = nn.Linear(64, 1)
        self.sigma = nn.Linear(64, 1)
    
    def forward(self, t, x):  # Fixed: takes both time and state as input
        inputs = torch.cat([t, x], dim=1)
        h = self.hidden(inputs)
        mu = self.mu(h)
        sigma = torch.exp(self.sigma(h))  # Ensure sigma is positive
        return mu, sigma

model = DriftDiffusionNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 1000
dt = t[1] - t[0]

for epoch in range(n_epochs):
    total_loss = 0
    optimizer.zero_grad()  # Move optimizer.zero_grad() outside inner loop
    
    for i in range(1, len(t)):
        t_input = t_tensor[i-1]
        x_prev = x_tensor[i-1]
        x_target = x_tensor[i]
        
        mu, sigma = model(t_input, x_prev)  # Fixed: pass both t and x
        
        # Calculate expected next value (deterministic part)
        x_pred_mean = x_prev + mu * dt
        
        # Loss based on the drift term (more stable than using random noise)
        loss = loss_fn(x_pred_mean, x_target)
        total_loss += loss.item()
    
    # Backprop after accumulating losses from all time steps
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / (len(t) - 1)}")

# Simulate from the learned model
model.eval()
simulated = [torch.tensor([[1.0]], device=device)]

with torch.no_grad():  # Added: disable gradients for inference
    for i in range(1, len(t)):
        t_input = t_tensor[i-1]
        x_prev = simulated[-1]
        
        mu, sigma = model(t_input, x_prev)
        noise = torch.randn_like(mu)
        x_next = x_prev + mu * dt + sigma * np.sqrt(dt) * noise
        simulated.append(x_next)

simulated = torch.cat(simulated).detach().cpu().numpy()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, x, label='True OU path', alpha=0.8)
plt.plot(t, simulated, label='Learned SDE path', linestyle='--', alpha=0.8)
plt.legend()
plt.title("Ornstein-Uhlenbeck Process: True vs Learned")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.show()

# Additional: Plot the learned drift and diffusion functions
with torch.no_grad():
    t_plot = torch.linspace(0, 10, 100).unsqueeze(1).to(device)
    x_plot = torch.linspace(-2, 3, 100).unsqueeze(1).to(device)
    
    # Create meshgrid for visualization
    T_mesh, X_mesh = torch.meshgrid(t_plot.squeeze(), x_plot.squeeze(), indexing='ij')
    T_flat = T_mesh.reshape(-1, 1)
    X_flat = X_mesh.reshape(-1, 1)
    
    mu_pred, sigma_pred = model(T_flat, X_flat)
    mu_pred = mu_pred.reshape(T_mesh.shape).cpu().numpy()
    sigma_pred = sigma_pred.reshape(T_mesh.shape).cpu().numpy()

# Plot the learned drift function
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Drift function
im1 = ax1.contourf(T_mesh.cpu(), X_mesh.cpu(), mu_pred, levels=20, cmap='RdBu_r')
ax1.set_xlabel('Time')
ax1.set_ylabel('State')
ax1.set_title('Learned Drift Function μ(t,x)')
plt.colorbar(im1, ax=ax1)

# Diffusion function
im2 = ax2.contourf(T_mesh.cpu(), X_mesh.cpu(), sigma_pred, levels=20, cmap='viridis')
ax2.set_xlabel('Time') 
ax2.set_ylabel('State')
ax2.set_title('Learned Diffusion Function σ(t,x)')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
