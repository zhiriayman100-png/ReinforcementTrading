import streamlit as st
import pandas as pd
import numpy as np
import time
import tempfile
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
window_size = st.sidebar.number_input("Window size", value=30, min_value=5, step=1)
sl_options = st.sidebar.multiselect("SL options (pips)", options=[5,10,15,25,30,60,90,120], default=[15,30])
tp_options = st.sidebar.multiselect("TP options (pips)", options=[5,10,15,25,30,60,90,120], default=[30,60])
spread_pips = st.sidebar.number_input("Spread (pips)", value=1.0, step=0.1)
commission_pips = st.sidebar.number_input("Commission (pips)", value=0.0, step=0.1)
max_slippage_pips = st.sidebar.number_input("Max slippage (pips)", value=0.2, step=0.01)
open_penalty_pips = st.sidebar.number_input("Open penalty (pips)", value=0.5, step=0.1)
time_penalty_pips = st.sidebar.number_input("Time penalty (pips)", value=0.02, step=0.01)
hold_reward_weight = st.sidebar.number_input("Hold reward weight", value=0.005, step=0.001)
unrealized_delta_weight = st.sidebar.number_input("Unrealized delta weight", value=0.02, step=0.01)
allow_flip = st.sidebar.checkbox("Allow flip (close+open)", value=False)

# --- training controls ---
st.sidebar.header("Training")
model_type = st.sidebar.selectbox("Model", ("PPO", "DQN"))
episodes = st.sidebar.number_input("Number of episodes (optional)", value=0, min_value=0, step=1)
episode_max_steps = st.sidebar.number_input("Episode max steps", value=1000, min_value=10, step=10)
total_timesteps = st.sidebar.number_input("Total timesteps (override)", value=20000, step=1000)
ckpt_freq = st.sidebar.number_input("Checkpoint save frequency", value=50000, step=1000)

col1, col2 = st.columns([2,1])
with col1:
    st.write("## Dataset & Preview")
    if data_choice:
        st.write(f"Using dataset: {data_choice}")
        try:
            df_preview = pd.read_csv(data_choice, nrows=200)
            st.dataframe(df_preview)
        except Exception as e:
            st.error(f"Could not preview dataset: {e}")
    else:
        st.info("No dataset chosen yet.")

with col2:
    st.write("## Controls")
    if st.button("Train"):
        if not data_choice:
            st.error("Please provide a dataset (upload / select / generate).")
        else:
            st.info(f"Using dataset: {data_choice}")
            with st.spinner("Loading data and preprocessing..."):
                df, feature_cols = load_and_preprocess_data(data_choice)
                # compute timesteps
                if episodes > 0:
                    timesteps = int(episodes * episode_max_steps)
                else:
                    timesteps = int(total_timesteps)
                st.write(f"Training with {timesteps} total timesteps (model={model_type})")
            with st.spinner("Training — this may take a while..."):
                results = train_and_evaluate(
                    df=df,
                    feature_cols=feature_cols,
                    model_type=model_type,
                    total_timesteps=timesteps,
                    ckpt_save_freq=int(ckpt_freq),
                    win=int(window_size),
                    sl_opts=list(map(int, sl_options)),
                    tp_opts=list(map(int, tp_options)),
                    spread_pips=float(spread_pips),
                    commission_pips=float(commission_pips),
                    max_slippage_pips=float(max_slippage_pips),
                    random_start=True,
                    min_episode_steps=100,
                    episode_max_steps=int(episode_max_steps) if episode_max_steps > 0 else None,
                    hold_reward_weight=float(hold_reward_weight),
                    open_penalty_pips=float(open_penalty_pips),
                    time_penalty_pips=float(time_penalty_pips),
                    unrealized_delta_weight=float(unrealized_delta_weight),
                )

            st.success("Training completed")

            st.subheader("Metrics")
            st.write(f"Final equity (train): ${results['final_equity_train']:.2f}")
            st.write(f"Final equity (test): ${results['final_equity_test']:.2f}")

            if Path(results["equity_plot"]).exists():
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



