# Environment (env) — README

This folder contains the trading environment used during training and evaluation.

Key file

- `trading_env.py` — a Gym-like environment (`ForexTradingEnv`) that wraps forex OHLCV data and exposes an observation space, action space, and reward function suitable for RL.

How it's set up

- Data: The environment expects a preprocessed DataFrame (`pandas.DataFrame`) with feature columns produced by `indicators.load_and_preprocess_data`.
- Windowing: `window_size` slices past bars as part of the observation.
- Actions: typically include open/close/do-nothing or discretized position sizes depending on configuration.
- Rewards: composed of trade outcomes (PnL), optional time penalties, open penalties and optional unrealized PnL weighting.

Configuration & customization

Most environment parameters are passed when the env is constructed. Common parameters you can adjust:

- `window_size` (int): number of past bars in the observation window
- `sl_options` / `tp_options` (list[int]): allowed stop-loss / take-profit choices expressed in pips
- `spread_pips` / `commission_pips` (float): trading costs
- `max_slippage_pips` (float): max slippage per execution
- `random_start` (bool): randomize episode start index (useful during training)
- `min_episode_steps` / `episode_max_steps` (int): episode length control
- `hold_reward_weight`, `open_penalty_pips`, `time_penalty_pips`, `unrealized_delta_weight` (float): reward shaping knobs

To customize the environment behavior:

- Edit `env/trading_env.py` to add/remove observation channels or change the reward function.
- To change the action space (e.g., add position sizes), update both `action_space` and the action-handling logic in `step()`.
- If you add new state features, update the preprocessing pipeline in `indicators.py` to include them.

Testing & debugging

- Create a small synthetic DataFrame and instantiate `ForexTradingEnv` directly to step through episodes and inspect `infos` for `equity_usd` and other diagnostics.
- Use deterministic settings (`random_start=False`) to get reproducible equity curves for evaluation.

Notes

- The environment is designed to work inside a `DummyVecEnv` from Stable Baselines 3 for training and evaluation.