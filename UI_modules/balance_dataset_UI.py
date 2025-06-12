import streamlit as st
import pandas as pd
from balance_data import balance_and_export_dataset
from logger import log_action

def render_balance_dataset_ui():
    st.markdown("---")
    st.subheader("⚙️ Preprocess & Balance Dataset")
    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        Balances dataset across age groups via undersampling.
        Saves:
        - `final_output/balanced_annotations.csv`
        - updated frame annotations
        - exported video set
        """)

    if st.button("⚖️ Balance Dataset using Undersampling"):
        result = balance_and_export_dataset()
        st.session_state.balance_result = result
        st.session_state.balance_done = result["status"] == "success"

    if st.session_state.get("balance_done", False):
        result = st.session_state.balance_result
        st.markdown("🔍 Age-Distribution Before Balancing(undersampling)")
        st.dataframe(result["pre_distribution"])

        st.markdown("🔍 Age-Distribution After Balancing (undersampling)")
        st.dataframe(result["post_distribution"])

        if result["frame_annotated"]:
            st.success("🧾 Frame-level annotations saved.")

        st.success(f"📦 {result['copied']} videos exported to `{result['export_path']}`.")
        log_action("Data Balance", "SUCCESS", "Dataset balanced and exported.")
    elif st.session_state.get("balance_result", {}).get("status") == "error":
        for msg in st.session_state.balance_result.get("messages", []):
            st.error(msg)
