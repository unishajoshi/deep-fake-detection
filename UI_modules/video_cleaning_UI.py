import os
import streamlit as st
from cleanup import clean_low_quality_videos
from logger import log_action

def render_video_cleaning_ui():
    st.markdown("---")
    st.subheader("🧹🧼 Preprocess: Remove Low-Resolution Files")

    output_video_dir = "all_data_videos"

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Scans all videos in `all_data_videos/real` and `all_data_videos/fake`.
        - Identifies low-quality or unreadable videos (e.g., corrupt, 0-byte, unreadable by OpenCV).
        - Deletes videos that fail the quality check.
        """)

    if "clean_success" not in st.session_state:
        st.session_state.clean_success = False

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("🧹 **Clean Low-Quality Videos**"):
        if os.path.exists(output_video_dir):
            clean_low_quality_videos(output_video_dir)
            st.session_state.clean_success = True
            log_action("Data Cleaning", "SUCCESS", "Data cleaning was successful")
        else:
            st.error("❌ Invalid path to video folder.")
            st.session_state.clean_success = False
            log_action("Data Cleaning", "FAIL", "Data cleaning could not complete")

    if st.session_state.clean_success:
        st.success("✅ Low-quality videos removed.")