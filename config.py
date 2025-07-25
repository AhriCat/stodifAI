import os

# Use .env or environment variables
API_KEY = os.getenv("PHEMEX_API_KEY")
SECRET = os.getenv("PHEMEX_SECRET")

SYMBOL = 'uBTC/USD:USD'
POSITION_SIZE = 10
THRESHOLD = 0.02
SLEEP_TIME = 10  # seconds
