import os
import streamlit as st
from evaluation import evaluate_on_all_sets, flatten_results_grouped
from logger import log_action
import pandas as pd

def render_model_evaluation_ui():
    st.markdown("### 📊 Evaluate Models on Balanced, Colab, FF++")

    if "eval_df" not in st.session_state:
        st.session_state.eval_df = None
    if "eval_done" not in st.session_state:
        st.session_state.eval_done = False

    if st.button("Run Evaluation"):
        if not st.session_state.get("selected_models"):
            st.warning("⚠️ Please train models first.")
            st.session_state.eval_done = False
        else:
            with st.spinner("Evaluating..."):
                results = evaluate_on_all_sets(
                    selected_models=st.session_state.selected_models,
                    streamlit_mode=True
                )
            st.session_state.eval_df = flatten_results_grouped(results)
            st.session_state.eval_done = True
            os.makedirs("final_output", exist_ok=True)
            st.session_state.eval_df.to_csv("final_output/evaluation_results.csv", index=False)
            log_action("Model Evaluation", "SUCCESS", "Evaluation results saved.")

    if st.session_state.eval_done and st.session_state.eval_df is not None:
        st.dataframe(st.session_state.eval_df, use_container_width=True)
        st.download_button(
            label="⬇️ Download Evaluation Summary",
            data=st.session_state.eval_df.to_csv(index=False),
            file_name="evaluation_results.csv",
            mime="text/csv"
        )