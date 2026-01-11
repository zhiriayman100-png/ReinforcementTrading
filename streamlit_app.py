import streamlit as st
import pandas as pd
import numpy as np
import time
import tempfile
import io
from pathlib import Path

import sys
from pathlib import Path

# Ensure repository root is on sys.path so local modules (e.g. train_agent)
# can be imported even when Streamlit changes the working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# If the script is executed directly with `python3 streamlit_app.py`, print
# a friendly instruction and exit early. When run by `streamlit run`, the
# Streamlit runner provides a script run context and we should NOT exit.
try:
    ctx = None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        try:
            from streamlit.scriptrunner.script_run_context import get_script_run_ctx
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None
    if __name__ == "__main__" and ctx is None:
        print("Please run this app using Streamlit: 'streamlit run streamlit_app.py'")
        print("For headless/devcontainer: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true")
        sys.exit(0)
except Exception:
    # If detection fails for any reason, fall back to not blocking execution.
    pass

import importlib.util

# Try normal imports first; if they fail under Streamlit's runner, load by path
try:
    from indicators import load_and_preprocess_data
except Exception:
    spec = importlib.util.spec_from_file_location("indicators", str(ROOT / "indicators.py"))
    indicators = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indicators)
    load_and_preprocess_data = indicators.load_and_preprocess_data

try:
    from train_agent import train_and_evaluate
except Exception:
    spec = importlib.util.spec_from_file_location("train_agent", str(ROOT / "train_agent.py"))
    train_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_agent)
    train_and_evaluate = train_agent.train_and_evaluate

st.set_page_config(page_title="RL Trading Trainer", layout="wide")
st.title("Reinforcement Trading — Train an Agent")

# --- Dataset controls ---
st.sidebar.header("Dataset")
dataset_mode = st.sidebar.radio("Dataset", ("Use existing CSV", "Upload CSV", "Generate synthetic"))

data_choice = None
if dataset_mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV data (OHLCV)", type=["csv"])
    if uploaded_file:
        tmpdir = Path("uploads")
        tmpdir.mkdir(exist_ok=True)
        tmp_path = tmpdir / uploaded_file.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_choice = str(tmp_path)
elif dataset_mode == "Use existing CSV":
    csvs = [str(p) for p in Path("data").glob("*.csv")]
    data_choice = st.sidebar.selectbox("Select CSV", options=csvs)
else:
    st.sidebar.write("Generate a synthetic OHLC dataset (simple random-walk)")
    n_bars = st.sidebar.number_input("Bars (synthetic)", value=3000, min_value=200, step=100)
    vol = st.sidebar.number_input("Volatility (approx)", value=0.0005, step=0.0001, format="%.6f")
    if st.sidebar.button("Generate Data"):
        def gen_synth(n, start_price=1.1000, vol=0.0005):
            np.random.seed(42)
            price = start_price
            rows = []
            for i in range(n):
                ret = np.random.normal(scale=vol)
                close = price * (1 + ret)
                high = max(price, close) * (1 + abs(np.random.normal(scale=vol/2)))
                low = min(price, close) * (1 - abs(np.random.normal(scale=vol/2)))
                openp = price
                volume = int(100 + abs(np.random.normal(scale=50))*100)
                rows.append([openp, high, low, close, volume])
                price = close
            df = pd.DataFrame(rows, columns=["Open","High","Low","Close","Volume"])
            tmp = Path(tempfile.gettempdir()) / f"synth_{int(time.time())}.csv"
            df.to_csv(tmp, index=False)
            return str(tmp)
        data_choice = gen_synth(int(n_bars), vol=vol)
        st.success(f"Generated synthetic CSV: {data_choice}")

# --- Env settings ---
st.sidebar.header("Env parameters")
window_size = st.sidebar.number_input(
    "Window size",
    value=30,
    min_value=5,
    step=1,
    help="Number of past bars included in each observation (larger -> more history, slower).")
sl_options = st.sidebar.multiselect(
    "SL options (pips)",
    options=[5,10,15,25,30,60,90,120],
    default=[15,30],
    help="Stop-loss choices (in pips) the agent may use when opening a trade.")
tp_options = st.sidebar.multiselect(
    "TP options (pips)",
    options=[5,10,15,25,30,60,90,120],
    default=[30,60],
    help="Take-profit choices (in pips) available to the agent.")
spread_pips = st.sidebar.number_input(
    "Spread (pips)",
    value=1.0,
    step=0.1,
    help="Market spread in pips — acts as a cost on each round-trip trade.")
commission_pips = st.sidebar.number_input(
    "Commission (pips)",
    value=0.0,
    step=0.1,
    help="Per-trade commission in pips (if your broker charges one).")
max_slippage_pips = st.sidebar.number_input(
    "Max slippage (pips)",
    value=0.2,
    step=0.01,
    help="Maximum slippage expected on order execution (simulated).")
open_penalty_pips = st.sidebar.number_input(
    "Open penalty (pips)",
    value=0.5,
    step=0.1,
    help="Penalty applied when opening a position to discourage excessive trading.")
time_penalty_pips = st.sidebar.number_input(
    "Time penalty (pips)",
    value=0.02,
    step=0.01,
    help="Per-step time penalty (in pips) to encourage shorter trades if desired.")
hold_reward_weight = st.sidebar.number_input(
    "Hold reward weight",
    value=0.005,
    step=0.001,
    help="Weight for holding reward component (small positive encourages holding profitable positions).")
unrealized_delta_weight = st.sidebar.number_input(
    "Unrealized delta weight",
    value=0.02,
    step=0.01,
    help="Weight applied to unrealized profit/loss when shaping rewards.")
allow_flip = st.sidebar.checkbox(
    "Allow flip (close+open)",
    value=False,
    help="Allow an immediate close and open in a single action (useful when changing position direction).")

# --- training controls ---
st.sidebar.header("Training")
model_type = st.sidebar.selectbox(
    "Model",
    ("PPO", "DQN"),
    help="Algorithm: PPO (on-policy) or DQN (off-policy). PPO is usually more stable for continuous tasks.")
episodes = st.sidebar.number_input(
    "Number of episodes (optional)",
    value=0,
    min_value=0,
    step=1,
    help="If >0, the app will compute total timesteps as episodes * episode_max_steps.")
episode_max_steps = st.sidebar.number_input(
    "Episode max steps",
    value=1000,
    min_value=10,
    step=10,
    help="Maximum number of steps per episode used when computing timesteps from episodes.")
total_timesteps = st.sidebar.number_input(
    "Total timesteps (override)",
    value=20000,
    step=1000,
    help="Total environment interaction steps for training (overrides episodes if episodes==0).")
ckpt_freq = st.sidebar.number_input(
    "Checkpoint save frequency",
    value=50000,
    step=1000,
    help="Save checkpoints every N timesteps so you can evaluate intermediate models.")

# --- Session state for multi-phase flow ---
if "phase" not in st.session_state:
    # phase: 1=data select, 2=train config, 3=results
    st.session_state.phase = 1

if "df" not in st.session_state:
    st.session_state.df = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "results" not in st.session_state:
    st.session_state.results = None

# safe rerun helper: some streamlit versions don't expose experimental_rerun
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        return

# Top-level navigation (full-page sections)
if "tab" not in st.session_state:
    st.session_state.tab = "Data"

try:
    # newer streamlit supports horizontal radios
    tab = st.radio("", ("Data", "Train", "Results"), index=(0 if st.session_state.tab == "Data" else 1 if st.session_state.tab == "Train" else 2), horizontal=True)
except TypeError:
    tab = st.radio("", ("Data", "Train", "Results"), index=(0 if st.session_state.tab == "Data" else 1 if st.session_state.tab == "Train" else 2))
st.session_state.tab = tab

# --- Data tab ---
if st.session_state.tab == "Data":
    st.header("Dataset & Preview")
    st.write("Use the dataset controls (left panel) or upload/select/generate below.")

    with st.expander("Data format & example (click for details)"):
        st.markdown(
            """
            **Expected CSV format**

            - **Datetime column**: one of `Time (EET)`, `Gmt time`, or `Time` (the loader detects these names).
            - **OHLCV columns**: `Open`, `High`, `Low`, `Close`, `Volume` (numeric).
            - Dates are parsed with day-first by default — ISO-like datetimes such as `2023-02-20 12:00` are supported.

            The preprocessing pipeline (`indicators.load_and_preprocess_data`) computes technical features and drops initial NaNs. If your CSV has a slightly different header (e.g. trailing spaces), the loader will try to handle it, but ensure the core columns exist.
            """
        )
        # provide a downloadable sample CSV
        sample_df = pd.DataFrame({
            "Time": pd.date_range("2025-01-01", periods=10, freq="h"),
            "Open": [1.10 + i * 0.0001 for i in range(10)],
            "High": [1.1005 + i * 0.0001 for i in range(10)],
            "Low": [1.0995 + i * 0.0001 for i in range(10)],
            "Close": [1.1002 + i * 0.0001 for i in range(10)],
            "Volume": [100 + i * 10 for i in range(10)],
        })
        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download sample CSV", data=csv_bytes, file_name="sample_eurusd.csv", mime="text/csv")

    # Repeat dataset controls (kept minimal to avoid changing logic)
    dataset_mode_local = st.radio("Dataset", ("Use existing CSV", "Upload CSV", "Generate synthetic"))
    data_choice_local = None
    if dataset_mode_local == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV data (OHLCV)", type=["csv"]) 
        if uploaded_file:
            tmpdir = Path("uploads")
            tmpdir.mkdir(exist_ok=True)
            tmp_path = tmpdir / uploaded_file.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_choice_local = str(tmp_path)
    elif dataset_mode_local == "Use existing CSV":
        csvs = [str(p) for p in Path("data").glob("*.csv")]
        data_choice_local = st.selectbox("Select CSV", options=csvs)
    else:
        st.write("Generate a synthetic OHLC dataset (simple random-walk)")
        n_bars_local = st.number_input("Bars (synthetic)", value=3000, min_value=200, step=100)
        vol_local = st.number_input("Volatility (approx)", value=0.0005, step=0.0001, format="%.6f")
        if st.button("Generate Data"):
            def gen_synth(n, start_price=1.1000, vol=0.0005):
                np.random.seed(42)
                price = start_price
                rows = []
                for i in range(n):
                    ret = np.random.normal(scale=vol)
                    close = price * (1 + ret)
                    high = max(price, close) * (1 + abs(np.random.normal(scale=vol/2)))
                    low = min(price, close) * (1 - abs(np.random.normal(scale=vol/2)))
                    openp = price
                    volume = int(100 + abs(np.random.normal(scale=50))*100)
                    rows.append([openp, high, low, close, volume])
                    price = close
                df = pd.DataFrame(rows, columns=["Open","High","Low","Close","Volume"])
                tmp = Path(tempfile.gettempdir()) / f"synth_{int(time.time())}.csv"
                df.to_csv(tmp, index=False)
                return str(tmp)
            data_choice_local = gen_synth(int(n_bars_local), vol=vol_local)
            st.success(f"Generated synthetic CSV: {data_choice_local}")

    # Decide which dataset to use (priority: local selection, else earlier sidebar selection)
    effective_data = data_choice_local if data_choice_local else data_choice

    if effective_data:
        st.write(f"Using dataset: {effective_data}")
        try:
            df_preview = pd.read_csv(effective_data, nrows=200)
            st.dataframe(df_preview)
        except Exception as e:
            st.error(f"Could not preview dataset: {e}")
    else:
        st.info("No dataset chosen yet.")

    if st.button("Prepare for Training"):
        if not effective_data:
            st.error("Please provide a dataset (upload / select / generate).")
        else:
            st.info(f"Preprocessing dataset: {effective_data}")
            with st.spinner("Loading data and preprocessing..."):
                try:
                    df, feature_cols = load_and_preprocess_data(effective_data)
                    st.session_state.df = df
                    st.session_state.feature_cols = feature_cols
                    st.session_state.phase = 2
                    st.session_state.tab = "Train"
                    safe_rerun()
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

# --- Train tab ---
if st.session_state.tab == "Train":
    st.header("Training")
    # Phase navigation
    if st.session_state.phase == 1:
        st.info("Trainer is waiting for a dataset. Prepare your dataset in the Data tab.")
    elif st.session_state.phase == 2:
        st.success("Ready to configure training parameters and start training.")
    elif st.session_state.phase == 3:
        st.success("Training finished — view results or retrain.")

    st.markdown("---")

    # Show current dataset
    st.write("**Current dataset**")
    if st.session_state.df is not None:
        st.write("(dataset preprocessed)")
    else:
        st.write("(none) — preprocess a dataset first")

    with st.expander("Parameter help (click for descriptions)"):
        st.markdown(
            """
            **Model**: Algorithm to use. *PPO* (on-policy, often stable), *DQN* (off-policy Q-learning).

            **Number of episodes**: Optional. If >0, total timesteps = episodes * episode_max_steps.

            **Episode max steps**: Maximum steps per episode used when calculating timesteps from episodes.

            **Total timesteps**: Total environment interactions to train the model.

            **Checkpoint save frequency**: Save intermediate model checkpoints every N timesteps.

            **Environment knobs**: Window size controls how much history the agent sees; spread/commission/slippage simulate costs; penalties and weights are reward-shaping parameters.
            """
        )

    # Training configuration (move inputs here so Train occupies full page)
    model_choice = st.selectbox("Model", ("PPO", "DQN"), index=(0 if model_type == "PPO" else 1))
    episodes_choice = st.number_input("Number of episodes (optional)", value=int(episodes), min_value=0, step=1, help="If set, total timesteps will be computed from episodes * episode_max_steps.")
    episode_max_steps_choice = st.number_input("Episode max steps", value=int(episode_max_steps), min_value=1, step=1, help="Maximum steps per episode used when computing total timesteps from episodes.")
    total_timesteps_choice = st.number_input("Total timesteps (override)", value=int(total_timesteps), step=1000, help="Total steps to train; used if episodes == 0.")
    ckpt_freq_choice = st.number_input("Checkpoint save frequency", value=int(ckpt_freq), step=1000, help="Save model checkpoints every N timesteps.")

    st.write("### Environment summary")
    st.write(f"Window size: {window_size}")
    st.write(f"SL options: {sl_options}")
    st.write(f"TP options: {tp_options}")
    st.write(f"Spread: {spread_pips} pips")

    st.markdown("---")
    col_start, col_back = st.columns([1, 1])
    with col_start:
        if st.button("Start Training"):
            if st.session_state.df is None or st.session_state.feature_cols is None:
                st.error("No preprocessed data available, please prepare the dataset first.")
            else:
                with st.spinner("Training — this may take a while..."):
                    # compute timesteps
                    if int(episodes_choice) > 0:
                        timesteps = int(episodes_choice * int(episode_max_steps_choice))
                    else:
                        timesteps = int(total_timesteps_choice)

                    results = train_and_evaluate(
                        df=st.session_state.df,
                        feature_cols=st.session_state.feature_cols,
                        model_type=model_choice,
                        total_timesteps=timesteps,
                        ckpt_save_freq=int(ckpt_freq_choice),
                        win=int(window_size),
                        sl_opts=list(map(int, sl_options)),
                        tp_opts=list(map(int, tp_options)),
                        spread_pips=float(spread_pips),
                        commission_pips=float(commission_pips),
                        max_slippage_pips=float(max_slippage_pips),
                        random_start=True,
                        min_episode_steps=100,
                        episode_max_steps=int(episode_max_steps_choice) if episode_max_steps_choice > 0 else None,
                        hold_reward_weight=float(hold_reward_weight),
                        open_penalty_pips=float(open_penalty_pips),
                        time_penalty_pips=float(time_penalty_pips),
                        unrealized_delta_weight=float(unrealized_delta_weight),
                    )

                st.session_state.results = results
                st.session_state.phase = 3
                st.session_state.tab = "Results"
                safe_rerun()
    with col_back:
        if st.button("Back to Data"):
            st.session_state.phase = 1
            st.session_state.tab = "Data"
            safe_rerun()

# --- Results tab ---
if st.session_state.tab == "Results" and st.session_state.results is not None:
    results = st.session_state.results
    st.header("Results")
    st.subheader("Metrics")
    st.write(f"Final equity (train): ${results['final_equity_train']:.2f}")
    st.write(f"Final equity (test): ${results['final_equity_test']:.2f}")

    if Path(results.get("equity_plot", "")).exists():
        st.subheader("Equity plot")
        st.image(results["equity_plot"], use_column_width=True)

    st.subheader("Artifacts")
    if results.get("model_zip") and Path(results.get("model_zip")).exists():
        with open(results["model_zip"], "rb") as f:
            st.download_button("Download model (.zip)", data=f, file_name=Path(results["model_zip"]).name)
    if results.get("meta_json") and Path(results.get("meta_json")).exists():
        with open(results["meta_json"], "rb") as f:
            st.download_button("Download meta (.json)", data=f, file_name=Path(results["meta_json"]).name)

    st.subheader("Equity curves (inline)")
    df_plot = pd.DataFrame({"train": results["equity_curve_train"], "test": results["equity_curve_test"]}).fillna(method="ffill")
    st.line_chart(df_plot)

    st.markdown("---")
    col_rerun, col_new = st.columns([1, 1])
    with col_rerun:
        if st.button("Retrain (change params)"):
            st.session_state.phase = 2
            st.session_state.tab = "Train"
            safe_rerun()
    with col_new:
        if st.button("Select new data"):
            st.session_state.phase = 1
            st.session_state.df = None
            st.session_state.feature_cols = None
            st.session_state.results = None
            st.session_state.tab = "Data"
            safe_rerun()



