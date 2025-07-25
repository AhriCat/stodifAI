import torch
from train import train_model
from utils import simulate_trajectory, plot_trajectories
from trade import run_trading_loop

def main(train_only=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, t_tensor, dt = train_model(device)
    simulated = simulate_trajectory(model, t_tensor, dt, device)

    from data import generate_ou_data
    t, x = generate_ou_data()
    plot_trajectories(t, x, simulated)

    if not train_only:
        run_trading_loop(model, t_tensor, dt, device)

if __name__ == "__main__":
    main(train_only=True)  # Set to False to launch trading
