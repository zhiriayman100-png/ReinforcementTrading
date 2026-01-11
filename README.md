# Reinforcement Trading — RL Trainer

A compact research playground for training Reinforcement Learning (RL) agents to trade forex (EURUSD). The project provides:

- Data loading & preprocessing utilities
- A vectorized Gym-like trading environment in `env/`
- Agent wrappers for Stable Baselines 3 (`agents/`) (PPO, DQN)
- A training wrapper `train_agent.py` that runs training, checkpointing and evaluation
- A Streamlit UI (`streamlit_app.py`) for interactive dataset selection, training and results

Quick start

1. Install requirements (use a virtualenv):

   pip install -r Requirements.txt

2. Run the Streamlit UI:

   streamlit run streamlit_app.py

3. Or run training from the command line:

   python3 train_agent.py

Where to look next

- `env/` — the trading environment implementation and configuration (see `env/README.md`)
- `agents/` — agent wrappers and examples (see `agents/README.md`)
- `data/` — example CSV files used for experiments

Contributing

Contributions are welcome. Open an issue or a PR with a clear description and, if applicable, a reproducible example.

License

This project is provided as-is for research and experimentation. Check the repository LICENSE if present.