import os
import streamlit as st
from frame_extractor import extract_frames_from_combined_parallel
from logger import log_action

def render_frame_extraction_ui():
    st.markdown("---")
    st.subheader("🖼️ Extract Frames")

    with st.expander("ℹ️ Frame Extraction Logic"):
        st.markdown("""
        - Combines uploaded real and fake videos into a single folder.
        - Extracts frames using OpenCV and parallel processing.
        - Saves frames as `.jpg` in `all_data_frames/`.
        """)

    frame_rate = st.selectbox("Select frame extraction rate (every Nth frame)", options=[1, 2, 5, 10, 15, 20, 25, 30], index=3)
    overwrite = st.checkbox("Overwrite existing frames if already extracted", value=False)

    if "frame_output_dir" not in st.session_state:
        st.session_state.frame_output_dir = None

    col1, col2, col3 = st.columns([2, 1, 1])
    if col3.button("🚀 Run Frame Extraction"):
        combined_dir = st.session_state.get("combined_dir", "all_data_videos")
        if combined_dir and os.path.exists(combined_dir):
            st.session_state.frame_output_dir = extract_frames_from_combined_parallel(
                combined_dir, frame_rate=frame_rate, overwrite=overwrite, streamlit_mode=True, batch_mode=False
            )
            st.session_state.frame_extraction_success = True
            st.success(f"✅ Frames extracted to: `{st.session_state.frame_output_dir}`")
            log_action("Frame Extraction", "SUCCESS", "Frames extracted.")
        else:
            st.error("❌ Combined video directory not found.")
            st.session_state.frame_extraction_success = False
            log_action("Frame Extraction", "FAIL", "Frame extraction failed.")