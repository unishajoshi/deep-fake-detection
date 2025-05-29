import streamlit as st
import pandas as pd
from model_trainer import prepare_data_split
from logger import log_action
##------------Train and Test Split-----------
# -------------------------
# 🔄 Train-Test Split
# -------------------------
def render_train_test_split_ui():
    st.markdown("### 🔄 Train-Test Split (70/30)")
    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        This step splits the **balanced dataset** into two sets:
        
        - **70% Training Set**: Used to train the deepfake detection models.
        - **30% Test Set**: Used to evaluate model performance on unseen data.
    
        The split is performed **after balancing the dataset by age groups** to ensure that all age ranges are fairly represented in both training and testing phases.
    
        🔁 This split is saved as:
        - `train_split.csv` (for training)
        - `test_split.csv` (for evaluation)
        """)
    # ✅ Initialize session state
    if "train_df" not in st.session_state:
        st.session_state.train_df = None
    if "test_df" not in st.session_state:
        st.session_state.test_df = None
    if "split_done" not in st.session_state:
        st.session_state.split_done = False
    
    # Button to trigger split
    if st.button("Perform Split"):
        from model_trainer import prepare_data_split
        st.session_state.train_df, st.session_state.test_df = prepare_data_split("frame_level_annotations.csv")
        st.session_state.split_done = True
    
        # ✅ Show persistent success message and preview
        if st.session_state.split_done and st.session_state.train_df is not None:
            st.success("✅ Split complete.")
            log_action("Train/Test Split", "SUCCESS", "Successfully splitted the balanced dataset for training and testing")
        else:
            st.error("❌ Split failed. Please check if the dataset is loaded and try again.")
            log_action("Train/Test Split", "FAIL", "Failed to splitted the balanced dataset for training and testing")
