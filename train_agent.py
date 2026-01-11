import os
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from env.trading_env import ForexTradingEnv
from agents.ppo_agent import PPOAgent
from config import MODELS_DIR, CKPT_DIR, TENSORBOARD_DIR
from utils.plotting import plot_equity_curves, plot_single_curve


def evaluate_model(model, eval_env, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def train_and_evaluate(
    df,
    feature_cols,
    model_type: str = "PPO",
    total_timesteps: int = 20000,
    ckpt_save_freq: int = 50000,
    win: int = 30,
    sl_opts: list | None = None,
    tp_opts: list | None = None,
    spread_pips: float = 1.0,
    commission_pips: float = 0.0,
    max_slippage_pips: float = 0.2,
    random_start: bool = True,
    min_episode_steps: int = 300,
    episode_max_steps: int | None = None,
    hold_reward_weight: float = 0.0,
    open_penalty_pips: float = 0.0,
    time_penalty_pips: float = 0.0,
    unrealized_delta_weight: float = 0.0,
    save_outputs: bool = True,
):
    """
    Parameterized training wrapper for programmatic calls (eg. Streamlit).
    Returns a dictionary with equity curves, final equities, plot & artifact paths.
    """
    if sl_opts is None:
        sl_opts = [5, 10, 15, 25, 30]
    if tp_opts is None:
        tp_opts = [5, 10, 15, 25, 30]

    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Env factories using passed params
    def make_env(random_start_local, df_for_env):
        return ForexTradingEnv(
            df=df_for_env,
            window_size=win,
            sl_options=sl_opts,
            tp_options=tp_opts,
            spread_pips=spread_pips,
            commission_pips=commission_pips,
            max_slippage_pips=max_slippage_pips,
            random_start=random_start_local,
            min_episode_steps=min_episode_steps,
            episode_max_steps=episode_max_steps,
            feature_columns=feature_cols,
            hold_reward_weight=hold_reward_weight,
            open_penalty_pips=open_penalty_pips,
            time_penalty_pips=time_penalty_pips,
            unrealized_delta_weight=unrealized_delta_weight,
        )

    from stable_baselines3.common.vec_env import DummyVecEnv

    train_vec_env = DummyVecEnv([lambda: make_env(True, train_df)])
    train_eval_env = DummyVecEnv([lambda: make_env(False, train_df)])
    test_eval_env = DummyVecEnv([lambda: make_env(False, test_df)])

    # ---- Agent selection ----
    model_type_up = model_type.upper()

    if model_type_up == "PPO":
        agent = PPOAgent.from_scratch(env=train_vec_env, policy="MlpPolicy", verbose=1, tensorboard_log=str(TENSORBOARD_DIR) + "/")
        ckpt_prefix = "ppo_eurusd"
    elif model_type_up == "DQN":
        from agents.dqn_agent import DQNAgent

        agent = DQNAgent.from_scratch(env=train_vec_env, policy="MlpPolicy", verbose=1,
                                      tensorboard_log=str(TENSORBOARD_DIR) + "/", buffer_size=50000)
        ckpt_prefix = "dqn_eurusd"
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    # ---- Checkpoints ----
    checkpoint_callback = CheckpointCallback(
        save_freq=int(ckpt_save_freq),
        save_path=str(CKPT_DIR),
        name_prefix=ckpt_prefix
    )

    # ---- Train ----
    agent.train(env=train_vec_env, total_timesteps=int(total_timesteps), callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    equity_curve_test_last, final_equity_test_last = evaluate_model(agent, test_eval_env)
    best_equity = final_equity_test_last
    best_agent = agent

    ckpts = sorted(
        [f for f in os.listdir(str(CKPT_DIR)) if f.endswith(".zip") and f.startswith(ckpt_prefix)],
        key=lambda x: os.path.getmtime(os.path.join(str(CKPT_DIR), x))
    )

    for ck in ckpts:
        ck_path = os.path.join(str(CKPT_DIR), ck)
        try:
            if model_type_up == "PPO":
                ck_agent = PPOAgent.load(ck_path, env=test_eval_env)
            else:
                from agents.dqn_agent import DQNAgent

                ck_agent = DQNAgent.load(ck_path, env=test_eval_env)

            _, final_eq = evaluate_model(ck_agent, test_eval_env)
            if final_eq > best_equity:
                best_equity = final_eq
                best_agent = ck_agent
        except Exception:
            # skip checkpoints that cannot be evaluated
            continue

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_dir = Path(MODELS_DIR) / f"{model_type_up.lower()}_eurusd_best_{ts}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model (SB3 will append .zip)
    model_path_base = model_dir / "model"
    best_agent.save(str(model_path_base))

    meta = {
        "model_type": model_type_up,
        "saved_at": ts,
        "file": str(model_path_base) + ".zip",
        "final_equity_test_last": float(final_equity_test_last),
        "best_oos_equity": float(best_equity) if best_equity != -np.inf else None,
        "total_timesteps": int(total_timesteps),
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Final evaluations ----
    equity_curve_train, final_equity_train = evaluate_model(best_agent, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_agent, test_eval_env)

    plot_path = model_dir / "equity.png"
    plot_equity_curves(
        equity_curve_train,
        equity_curve_test,
        title="Equity Curves: In-sample vs Out-of-sample (Best Model)",
        save_path=plot_path,
    )

    out = {
        "equity_curve_train": equity_curve_train,
        "equity_curve_test": equity_curve_test,
        "final_equity_train": final_equity_train,
        "final_equity_test": final_equity_test,
        "equity_plot": str(plot_path),
        "model_zip": str(model_path_base) + ".zip",
        "meta_json": str(model_dir / "meta.json"),
        "model_dir": str(model_dir),
    }

    return out


def main():
    # Prefer existing files in the `data/` directory. You can override by setting
    # the `DATA_CSV` environment variable to a path.
    default_candidates = [
        os.environ.get("DATA_CSV", ""),
        "data/EURUSD_Hourly_Ask_2015.12.01_2025.12.16.csv",
        "data/EURUSD_15 Mins_Ask_2020.12.06_2025.12.12.csv",
        "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
        "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv",
    ]

    file_path = None
    for candidate in default_candidates:
        if not candidate:
            continue
        if os.path.exists(candidate):
            file_path = candidate
            break

    if file_path is None:
        raise FileNotFoundError(f"No candidate CSV found. Checked: {default_candidates}")

    print(f"Using data file: {file_path}")
    df, feature_cols = load_and_preprocess_data(file_path)

    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training bars:", len(train_df))
    print("Testing bars :", len(test_df))

    # ---- Env factories ----
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,#0.05
            open_penalty_pips=0.0,      # 0.5 half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Train-eval env: deterministic start, NO random starts (so curve is stable/reproducible)
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,      # half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,      # half a pip per open
            time_penalty_pips=0.00,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # ---- Agent selection ----
    model_type = os.environ.get("MODEL", "PPO").upper()

    if model_type == "PPO":
        agent = PPOAgent.from_scratch(
            env=train_vec_env,
            policy="MlpPolicy",
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR) + "/"
        )
        ckpt_prefix = "ppo_eurusd"
    elif model_type == "DQN":
        from agents.dqn_agent import DQNAgent

        agent = DQNAgent.from_scratch(
            env=train_vec_env,
            policy="MlpPolicy",
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR) + "/",
            buffer_size=int(os.environ.get("DQN_BUFFER", 50000)),
        )
        ckpt_prefix = "dqn_eurusd"
    else:
        raise ValueError(f"Unsupported MODEL type: {model_type}")

    # ---- Checkpoints ----
    ckpt_dir = CKPT_DIR

    checkpoint_callback = CheckpointCallback(
        save_freq=int(os.environ.get("CKPT_SAVE_FREQ", 50_000)),
        save_path=str(ckpt_dir),
        name_prefix=ckpt_prefix
    )

    # ---- Train ----
    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", 600000))
    agent.train(env=train_vec_env, total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    equity_curve_test_last, final_equity_test_last = evaluate_model(agent, test_eval_env)
    print(f"[OOS Eval] Last model final equity: {final_equity_test_last:.2f}")

    best_equity = -np.inf
    best_path = None

    ckpts = sorted(
        [f for f in os.listdir(str(ckpt_dir)) if f.endswith(".zip") and f.startswith(ckpt_prefix)],
        key=lambda x: os.path.getmtime(os.path.join(str(ckpt_dir), x))
    )

    for ck in ckpts:
        ck_path = os.path.join(str(ckpt_dir), ck)
        try:
            if model_type == "PPO":
                ck_agent = PPOAgent.load(ck_path, env=test_eval_env)
            else:
                from agents.dqn_agent import DQNAgent

                ck_agent = DQNAgent.load(ck_path, env=test_eval_env)

            _, final_eq = evaluate_model(ck_agent, test_eval_env)
            print(f"[OOS Eval] {ck} -> final equity: {final_eq:.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    # Decide best model
    if best_path is None or final_equity_test_last >= best_equity:
        print("Using last model as best (by OOS final equity).")
        best_agent = agent
    else:
        print(f"Using best checkpoint: {best_path} (OOS final equity: {best_equity:.2f})")
        if model_type == "PPO":
            best_agent = PPOAgent.load(best_path, env=train_vec_env)
        else:
            from agents.dqn_agent import DQNAgent

            best_agent = DQNAgent.load(best_path, env=train_vec_env)

    ts = time.strftime("%Y%m%d_%H%M%S")
    model_dir = Path(MODELS_DIR) / f"{model_type.lower()}_eurusd_best_{ts}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model (SB3 will append .zip)
    model_path_base = model_dir / "model"
    best_agent.save(str(model_path_base))

    meta = {
        "model_type": model_type,
        "saved_at": ts,
        "file": str(model_path_base) + ".zip",
        "final_equity_test_last": float(final_equity_test_last),
        "best_oos_equity": float(best_equity) if best_equity != -np.inf else None,
        "total_timesteps": total_timesteps,
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Best model saved under: {model_dir}")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    equity_curve_train, final_equity_train = evaluate_model(best_agent, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_agent, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plot_path = model_dir / "equity.png"
    plot_equity_curves(
        equity_curve_train,
        equity_curve_test,
        title="Equity Curves: In-sample vs Out-of-sample (Best Model)",
        save_path=plot_path,
    )
    print(f"Saved equity plot to {str(plot_path)}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
