import matplotlib.pyplot as plt
import torch

def plot_trajectories(t, true_x, simulated):
    plt.figure(figsize=(12, 6))
    plt.plot(t, true_x, label='True OU path')
    plt.plot(t, simulated, '--', label='Learned SDE path')
    plt.title("Ornstein-Uhlenbeck Process: True vs Learned")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def simulate_trajectory(model, t_tensor, dt, device):
    simulated = [torch.tensor([[1.0]], device=device)]
    model.eval()
    with torch.no_grad():
        for i in range(1, len(t_tensor)):
            t_input = t_tensor[i-1]
            x_prev = simulated[-1]
            mu, sigma = model(t_input, x_prev)
            noise = torch.randn_like(mu)
            x_next = x_prev + mu * dt + sigma * (dt ** 0.5) * noise
            simulated.append(x_next)
    return torch.cat(simulated).detach().cpu().numpy()
