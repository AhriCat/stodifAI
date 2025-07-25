# Setup
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Export your API keys (or use a .env loader)
export PHEMEX_API_KEY="your_key_here"
export PHEMEX_SECRET="your_secret_here"

# Train and simulate
python main.py

# To trade (switch flag)
- python main.py --train_only=False  <-- or edit the script
