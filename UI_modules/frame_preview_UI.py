import os
import streamlit as st
from frame_preview import preview_sample_frames, display_grid_pair
from logger import log_action

def render_frame_preview_ui():
    st.markdown("---")
    st.subheader("🔍 Preview Sample Frames")

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Previews up to 9 real and 9 fake video frames.
        - Frames are extracted and shown in a grid view.
        """)

    real_video_dir = os.path.join("all_data_videos", "real")
    fake_video_dir = os.path.join("all_data_videos", "fake")

    if "preview_real_grid" not in st.session_state:
        st.session_state.preview_real_grid = []
    if "preview_fake_grid" not in st.session_state:
        st.session_state.preview_fake_grid = []
    if "preview_done" not in st.session_state:
        st.session_state.preview_done = False

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("🖼️ Preview Frames"):
        if os.path.exists(real_video_dir) and os.path.exists(fake_video_dir):
            real_grid, fake_grid = preview_sample_frames(real_video_dir, fake_video_dir, return_images=True)
            st.session_state.preview_real_grid = real_grid
            st.session_state.preview_fake_grid = fake_grid
            st.session_state.preview_done = True
            log_action("Preview Frame", "SUCCESS", "Previewed frames.")
        else:
            st.error("❌ Please upload and save real/fake videos first.")
            st.session_state.preview_done = False
            log_action("Preview Frame", "FAIL", "Frame preview failed.")

    if st.session_state.preview_done:
        display_grid_pair(st.session_state.preview_real_grid, st.session_state.preview_fake_grid)