import torch
import time
import ccxt
from config import API_KEY, SECRET, SYMBOL, POSITION_SIZE, THRESHOLD, SLEEP_TIME

def setup_exchange():
    return ccxt.phemex({
        'apiKey': API_KEY,
        'secret': SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
        }
    })

def run_trading_loop(model, t_tensor, dt, device):
    exchange = setup_exchange()
    symbol = SYMBOL
    print("Started trading loop...")

    while True:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            t_now = torch.tensor([[t_tensor[-1].item() + dt]], device=device)
            x_now = torch.tensor([[price / 1000.0]], device=device)  # normalization optional

            mu, _ = model(t_now, x_now)
            mu_val = mu.item()
            print(f"Price: {price}, μ: {mu_val}")

            positions = exchange.fetch_positions([symbol])
            pos = positions[0]
            amt = float(pos['contracts'])
            side = pos['side']

            if mu_val > THRESHOLD and amt == 0:
                exchange.create_market_buy_order(symbol, POSITION_SIZE)
                print("Long executed.")

            elif mu_val < -THRESHOLD and amt == 0:
                exchange.create_market_sell_order(symbol, POSITION_SIZE)
                print("Short executed.")

            elif abs(mu_val) < THRESHOLD and amt > 0:
                if side == 'long':
                    exchange.create_market_sell_order(symbol, amt)
                elif side == 'short':
                    exchange.create_market_buy_order(symbol, amt)
                print("Position exited.")

            time.sleep(SLEEP_TIME)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)
