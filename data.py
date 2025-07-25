import numpy as np

def generate_ou_data(theta=0.7, mu=0.0, sigma=0.3, x0=1.0, T=10.0, N=1000):
    dt = T / N
    t = np.linspace(0, T, N)
    x = np.zeros(N)
    x[0] = x0
    for i in range(1, N):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return t, x
