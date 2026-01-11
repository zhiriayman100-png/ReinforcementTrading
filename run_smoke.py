from indicators import load_and_preprocess_data
from env.trading_env import ForexTradingEnv
import numpy as np


def main():
    csv = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv"
    df, feature_cols = load_and_preprocess_data(csv)

    # Use a small slice to keep the smoke test quick
    df = df.iloc[:2000].reset_index()

    env = ForexTradingEnv(
        df=df,
        window_size=30,
        sl_options=[10, 15],
        tp_options=[10, 15],
        feature_columns=feature_cols,
        random_start=False,
        episode_max_steps=100,
    )

    obs = env.reset()
    print("Initial equity:", env.equity_usd)

    for i in range(10):
        action = env.action_space.sample()
        out = env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
        else:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        print(f"Step {i}: action={action}, reward={reward:.4f}, equity={info.get('equity_usd')}")
        if done:
            break


if __name__ == '__main__':
    main()
