import os
import streamlit as st
from video_importer import import_real_images
from logger import log_action

def render_video_upload_ui():
    st.sidebar.subheader("📤 Upload Videos from celeb-DF")
    st.session_state["real_videos_1"] = st.sidebar.file_uploader("REAL Videos (celeb-DF)", type=["mp4", "avi"], accept_multiple_files=True, key="real_1")
    st.session_state["fake_videos_1"] = st.sidebar.file_uploader("FAKE Videos (celeb-DF)", type=["mp4", "avi"], accept_multiple_files=True, key="fake_1")

    st.sidebar.markdown("---")

    st.sidebar.subheader("📤 Upload Videos from FaceForensics++")
    st.session_state["real_videos_2"] = st.sidebar.file_uploader("REAL Videos (FaceForensics++)", type=["mp4", "avi"], accept_multiple_files=True, key="real_2")
    st.session_state["fake_videos_2"] = st.sidebar.file_uploader("FAKE Videos (FaceForensics++)", type=["mp4", "avi"], accept_multiple_files=True, key="fake_2")
    
def render_import_videos_ui():
    if st.sidebar.button("📥 Import All Videos to Combined Folder"):
        log_action("Import Data", "START", "Initiated import process.")
        

        base_output_dir = "all_data_videos"
        real_dir = os.path.join(base_output_dir, "real")
        fake_dir = os.path.join(base_output_dir, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        for rv in st.session_state.get("real_videos_1", []):
            with open(os.path.join(real_dir, f"celeb_data_real_{rv.name}"), "wb") as f:
                f.write(rv.read())

        for fv in st.session_state.get("fake_videos_1", []):
            with open(os.path.join(fake_dir, f"celeb_data_fake_{fv.name}"), "wb") as f:
                f.write(fv.read())

        for rv in st.session_state.get("real_videos_2", []):
            with open(os.path.join(real_dir, f"faceforensics_data_real_{rv.name}"), "wb") as f:
                f.write(rv.read())

        for fv in st.session_state.get("fake_videos_2", []):
            with open(os.path.join(fake_dir, f"faceforensics_data_fake_{fv.name}"), "wb") as f:
                f.write(fv.read())

        st.sidebar.success("✅ Videos successfully imported to `/all_data_videos`.")
        log_action("Import Data", "SUCCESS", "Imported all uploaded videos to local folder.")

#--------------------------------import real images----------------------

def render_image_import_ui():
    st.sidebar.subheader("🖼️ Import Real Images From UTKFace")

    uploaded_images = st.sidebar.file_uploader(
        "Upload Real Images (Format: [age]_[gender]_[race]_[timestamp].jpg)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="real_image_upload"
    )
    if st.sidebar.button("📥 Import Images"):
        df, csv_path = import_real_images(
            uploaded_images, 
            image_save_dir="all_data_videos/real_images", 
            annotations_file="all_data_videos/annotations.csv"
        )
        if df is not None:
            st.sidebar.success(f"✅ {len(df)} images processed. Metadata saved to `{csv_path}`")
            log_action("Image Import", "SUCCESS", f"{len(df)} images saved to real_images/")
        else:
            st.sidebar.error("❌ No valid images were processed.")
            log_action("Image Import", "FAIL", "Image metadata parsing failed")
