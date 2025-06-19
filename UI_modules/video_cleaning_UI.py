import os
import streamlit as st
from clean_data import clean_low_quality_videos, clean_low_quality_images
from logger import log_action

def render_video_cleaning_ui():
    st.markdown("---")
    st.subheader("🧹🧼 Preprocess: Remove Low-Resolution Files")

    output_video_dir = "all_data_videos"
    image_dir = "all_data_videos/real_images"
    annotations_file = "all_data_videos/annotations.csv"

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Scans all videos in `all_data_videos/real` and `all_data_videos/fake`.
        - Scans all real images in `all_data_videos/real_images`.
        - Deletes low-resolution or unreadable files.
        - Updates `annotations.csv` to remove metadata for bad images.
        """)

    if "clean_success" not in st.session_state:
        st.session_state.clean_success = False

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("🧹 **Clean Low-Quality Files**"):
        if os.path.exists(output_video_dir):
            # Clean low-quality videos
            removed_real_videos = clean_low_quality_videos(os.path.join(output_video_dir, "real"))
            removed_fake_videos = clean_low_quality_videos(os.path.join(output_video_dir, "fake"))

            # Clean low-quality images
            removed_images = clean_low_quality_images(image_dir, annotations_file)

            st.session_state.clean_success = True

            log_action(
                "Data Cleaning", "SUCCESS",
                f"Removed {len(removed_real_videos)} real videos, {len(removed_fake_videos)} fake videos, and {len(removed_images)} images"
            )
        else:
            st.error("❌ Invalid path to video folder.")
            st.session_state.clean_success = False
            log_action("Data Cleaning", "FAIL", "Data cleaning could not complete")

    if st.session_state.clean_success:
        st.success("✅ Low-quality images and videos removed.")