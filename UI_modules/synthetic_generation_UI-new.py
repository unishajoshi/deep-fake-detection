import streamlit as st
import os
import pandas as pd
from logger import log_action
from synthetic_data_simswap import generate_synthetic_videos
from data_preprocessing import FAKE_BALANCE_TARGET, merge_synthetic_into_balanced_annotations, generate_synthetic_frame_annotations
import json

def render_synthetic_generation_ui():
    

    st.markdown("## 🎭 Generate Synthetic Deepfake Videos")
    
    # Initialize session flags once
    if "generation_started" not in st.session_state:
        st.session_state["generation_started"] = False
    if "stop_generation" not in st.session_state:
        st.session_state["stop_generation"] = False
    if "simswap_process" not in st.session_state:
        st.session_state["simswap_process"] = None

    # Stop button logic — render only if generation has started
    if st.session_state.get("generation_started", False):
        if st.button("🛑 Stop Generation"):
            st.session_state["stop_generation"] = True
            proc = st.session_state.get("simswap_process")
            if proc and proc.poll() is None:
                proc.terminate()
                st.warning("🛑 SimSwap process terminated by user.")

    if st.button("🎬 Prepare & Generate Synthetic Videos"):
        st.session_state["generation_started"] = True
        st.session_state["stop_generation"] = False
    
        if not os.path.exists("final_output/synthetic_allocation.csv"):
            st.error("❌ synthetic_allocation.csv not found. Please run balancing first.")
            return
    
        synthetic_df = pd.read_csv("final_output/synthetic_allocation.csv")
    
        if synthetic_df.empty:
            st.success("✅ No synthetic data required. All age groups are already balanced.")
            return
    
        plan_summary = synthetic_df["age_group"].value_counts().sort_index()
        st.markdown("#### 🧠 Synthetic Generation Plan")
        st.dataframe(plan_summary.rename("Synthetic Videos to Create").reset_index().rename(columns={"index": "Age Group"}))
    
        progress_bar = st.progress(0, text="Generating synthetic videos...")
        generate_synthetic_videos(synthetic_df.reset_index(drop=True), streamlit_progress=progress_bar, st_module=st)
        frame_rate = st.session_state.get("selected_frame_rate", 30)
    
        merge_synthetic_into_balanced_annotations()
        generate_synthetic_frame_annotations(frame_rate=frame_rate)
    
        progress_bar.empty()
        st.success("✅ Synthetic videos generated and metadata updated.")
            
    