import os
import streamlit as st
import pandas as pd
from age_annotator import save_age_annotations_parallel
from data_preprocessing import generate_frame_level_annotations
from logger import log_action
import gc

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

    # Initialize session state
    st.session_state.setdefault("annotations_df", None)
    st.session_state.setdefault("age_annotation_done", False)

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("📋 Label Age Group"):
        if os.path.exists(video_annotation_dir) and os.path.exists(frame_dir):
            try:
                with st.spinner("Annotating age groups from video frames..."):
                    df = save_age_annotations_parallel(
                        video_dir=video_annotation_dir,
                        output_csv="all_data_videos/annotations.csv",
                        batch_mode=True,
                        num_workers=None,
                        streamlit_progress=None  # Disable progress bar for stability
                    )

                    # Avoid keeping large DataFrame in session_state if not needed
                    st.session_state.annotations_df = None  # Or store summary if desired
                    st.session_state.age_annotation_done = True

                    generate_frame_level_annotations(mode="annotated")
                    log_action("Age Annotation", "SUCCESS", "Age annotated.")

                    st.success("✅ Age annotation complete. Metadata saved to `annotations.csv`.")

            except Exception as e:
                st.error(f"❌ Age annotation failed: {e}")
                log_action("Age Annotation", "FAIL", f"Exception: {e}")
                st.session_state.age_annotation_done = False
        else:
            st.error("❌ Folder not found: `all_data_videos` or `all_data_frames`. Please upload videos first.")
            st.session_state.age_annotation_done = False
            log_action("Age Annotation", "FAIL", "Missing input folders.")

    # Clean up memory
    gc.collect()
