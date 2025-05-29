import os
import streamlit as st
import pandas as pd
from age_annotator import save_age_annotations_parallel
from data_preprocessing import generate_frame_level_annotations
from logger import log_action

def render_age_annotation_ui():
    st.markdown("---")
    st.subheader("🧓 Annotate and Save Age Groups")

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Detects age groups from frames using DeepFace.
        - Saves annotations to `annotations.csv`.
        """)

    video_annotation_dir = "all_data_videos"
    frame_dir = "all_data_frames"

    if "annotations_df" not in st.session_state:
        st.session_state.annotations_df = None
    if "age_annotation_done" not in st.session_state:
        st.session_state.age_annotation_done = False

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("📋 Label Age Group"):
        if os.path.exists(video_annotation_dir) and os.path.exists(frame_dir):
            progress_bar = st.progress(0)
            df = save_age_annotations_parallel(video_dir=video_annotation_dir, output_csv="all_data_videos/annotations.csv",
                                               batch_mode=True, num_workers=None, streamlit_progress=progress_bar)
            st.session_state.annotations_df = df
            st.session_state.age_annotation_done = True
            generate_frame_level_annotations(mode="annotated")
            log_action("Age Annotation", "SUCCESS", "Age annotated.")
            st.success("✅ Age annotation complete. Metadata saved to `annotations.csv`.")
        else:
            st.error("❌ Folder not found: `all_data_videos`. Please upload videos first.")
            st.session_state.age_annotation_done = False
            log_action("Age Annotation", "FAIL", "Age annotation failed.")