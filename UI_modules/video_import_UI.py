import os
import streamlit as st
from video_importer import import_real_images, import_real_images_from_zip
from video_importer import load_images_from_dir
from logger import log_action
import tempfile
import zipfile

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

#---------------------------import zip image logic---------------
def render_zip_import_ui():
    st.sidebar.subheader("🖼️ Import Real Images From UTKFace")

    uploaded_zip = st.sidebar.file_uploader(
        "Upload a ZIP File of Real Images (Format: [age]_[gender]_[race]_[timestamp].jpg)", 
        type=["zip"],
        key="real_image_zip_upload"
    )

    if "import_result" not in st.session_state:
        st.session_state.import_result = None

    if st.sidebar.button("📥 Import Images"):
        if uploaded_zip is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "uploaded_images.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                image_files = load_images_from_dir(temp_dir)

                df, csv_path = import_real_images(
                    image_files,
                    image_save_dir="all_data_videos/real_images",
                    annotations_file="all_data_videos/annotations.csv"
                )

                if df is not None:
                    st.session_state.import_result = {
                        "status": "success",
                        "message": f"✅ {len(df)} images processed. Metadata saved to `{csv_path}`"
                    }
                    log_action("Image Import", "SUCCESS", f"{len(df)} images saved to real_images/")
                else:
                    st.session_state.import_result = {
                        "status": "error",
                        "message": "❌ No valid images were processed."
                    }
                    log_action("Image Import", "FAIL", "Image metadata parsing failed")
        else:
            st.session_state.import_result = {
                "status": "warning",
                "message": "⚠️ Please upload a ZIP file first."
            }

    # 🎯 Show persistent message
    result = st.session_state.import_result
    if result:
        if result["status"] == "success":
            st.sidebar.success(result["message"])
        elif result["status"] == "error":
            st.sidebar.error(result["message"])
        elif result["status"] == "warning":
            st.sidebar.warning(result["message"])