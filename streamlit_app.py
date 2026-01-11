# removed per user request: Streamlit app
# file cleared to remove previous Streamlit functionality

# --- Dataset selection ---
uploaded_file = st.file_uploader("Upload your CSV data (OHLCV)", type=["csv"])

data_choice = None
if uploaded_file is not None:
    tmpdir = Path("uploads")
    tmpdir.mkdir(exist_ok=True)
    tmp_path = tmpdir / uploaded_file.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_choice = str(tmp_path)
else:
    # list csv files in data/ dir
    data_dir = Path("data")
    csvs = [str(p) for p in data_dir.glob("*.csv")]

    data_choice = st.selectbox("Or select an existing CSV", options=csvs)

st.sidebar.header("Training options")
model_type = st.sidebar.selectbox("Model", options=["PPO"], index=0)
timesteps = st.sidebar.number_input("Total timesteps", value=20000, step=1000)
ckpt_freq = st.sidebar.number_input("Checkpoint save frequency", value=50000, step=1000)

if st.sidebar.button("Train"):
    if data_choice is None:
        st.error("Please upload or select a CSV dataset.")
    else:
        st.info(f"Using dataset: {data_choice}")
        with st.spinner("Training in progress — this may take a while for large timesteps..."):
            # Load and preprocess
            df, feature_cols = load_and_preprocess_data(data_choice)
            results = train_and_evaluate(
                df=df,
                feature_cols=feature_cols,
                total_timesteps=int(timesteps),
                ckpt_save_freq=int(ckpt_freq),
                save_outputs=True,
            )

        st.success("Training completed")

        # Show metrics
        st.subheader("Metrics")
        st.write(f"Final equity (train): ${results['final_equity_train']:.2f}")
        st.write(f"Final equity (test): ${results['final_equity_test']:.2f}")

        # Show equity plot
        if Path(results["equity_plot"]).exists():
            st.subheader("Equity plot")
            st.image(results["equity_plot"], use_column_width=True)

        # Allow downloads
        st.subheader("Artifacts")
        model_zip = results.get("model_zip")
        meta_json = results.get("meta_json")
        if model_zip and Path(model_zip).exists():
            with open(model_zip, "rb") as f:
                st.download_button("Download model (.zip)", data=f, file_name=Path(model_zip).name)
        if meta_json and Path(meta_json).exists():
            with open(meta_json, "rb") as f:
                st.download_button("Download meta (.json)", data=f, file_name=Path(meta_json).name)

        # Plot curves inline
        st.subheader("Equity Curves (inline)")
        df_plot = pd.DataFrame({
            "train": results["equity_curve_train"],
            "test": results["equity_curve_test"],
        }).fillna(method="ffill")
        st.line_chart(df_plot)

        st.write("Done.")

else:
    st.info("Configure training options in the sidebar and click Train to start.")
