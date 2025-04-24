# Streamlit & UI Utilities
import streamlit as st
from streamlit_modal import Modal
import shutil
from io import BytesIO
import zipfile
import logging
from datetime import datetime

# Create logs directory


# Data Handling & Visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Core Python & System
import os
os.makedirs("logs", exist_ok=True)
import torch
import asyncio
import torch.nn as nn


# Project Modules
from video_importer import combine_videos, extract_frames_from_combined_parallel, preview_sample_frames,display_grid_pair
from age_annotator import save_age_annotations_parallel
from data_preprocessing import balance_dataset, export_balanced_dataset,generate_frame_level_annotations
from visualization import (
    get_age_label_source_table,
    visualize_age_distribution,
    show_age_distribution_pie_charts
)
from cleanup import clean_import_directory, clean_low_quality_videos
from model_trainer import prepare_data_split, train_models, get_model, train_models_on_source
from evaluation import evaluate_model, evaluate_on_all_sets, flatten_results_grouped, flatten_age_specific_results,evaluate_on_all_sets_agewise, evaluate_on_all_sets_for_trained_models
from balance_data import balance_and_export_dataset
from grad_cam import apply_gradcam
from wrapup import create_full_pdf_report
from logger import log_action


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize Session State
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "train_df" not in st.session_state:
    st.session_state.train_df = None

if "test_df" not in st.session_state:
    st.session_state.test_df = None

    
#------------------ Section for: TITLE --------------
st.set_page_config(page_title="Deepfake Dataset Builder", layout="wide")
st.title("🧠 Age-Diverse Deepfake Dataset Pipeline")

#------------------ Section for: SIDEBAR --------------
with st.sidebar.expander("❓ Help & Instructions", expanded=False):
    st.markdown("""
    ### 🛠️ Application Functionality
    In this application we will create and evaluate an **age-diverse deepfake detection pipeline** in a step-by-step approach. The goal of this applicaiton is to generate the age-diverse deepfake dataset that can assist in the fairer deepfake detection system. The following functionalities will be coverd:

    1. **📤 Import Videos**
       - Imports real and fake videos from celeb-DF-v2 and FaceForensics++ datasets (from predefined local folders).
    
    2. **🧹 Preprocessing**
       - Removes low-quality or unreadable videos.
    
    3. **🖼️ Frame Extraction**
       - Extracts frames from all videos at a user-specified rate.
    
    4. **🔍 Preview Frames**
       - Shows visual previews of real and fake frames for quality check.
    
    5. **🧓 Age Annotation**
       - Uses face analysis to annotate each video with an estimated age group.

    6. **📊 Dataset Visualization**
       - Displays distributions of age groups and dataset sources in bar and pie chart formats.

    7. **⚖️ Balancing**
       - Balances the dataset across age groups using under/oversampling.

    8. **🔄 Train-Test Split**
       - Splits the balanced data into 70% training and 30% test sets.

    9. **🧠 Model Training**
       - Trains deepfake detection models: XceptionNet, EfficientNet, LipForensics.

    10. **📊 Evaluation**
        - Evaluates models on balanced, celeb, and FaceForensics++ datasets.
        - Supports age-group–specific performance metrics.
    
    11. **🧠 Model Training/Evaluation using Source dataset ( celeb and Face Forensics ++)**
        - Trains deepfake detection models: XceptionNet, EfficientNet, LipForensics. Used for compared performance between the created age-diverse deepfake dataset and the origial source datasets.

    12. **🎯 Grad-CAM**
        - Visualizes model attention via Grad-CAM heatmaps for real vs fake frames.

    13. **📄 PDF Reporting**
        - Generates a full project summary PDF report including results, charts, and samples.

    14. **📦 Dataset Export**
        - Exports balanced videos, annotations, and evaluation results as a ZIP file.
    """)

# ------------------ SECURITY DISCLAIMER ------------------
with st.sidebar.expander("🔐 Security Disclaimer", expanded=False):
    st.markdown("""
    ### 🔒 Data Privacy & Usage
    - The source videos (real and fake) are collected from publicly available deepfake datasets: **celeb-DF-v2** and **FaceForensics++**.
    - This tool is intended strictly for **research and educational use only**.
    - The app does **not expose or share raw video datasets**. All video and metadata processing happens **locally on the server-side session**.
    - Users can view dataset summaries, sample previews, and evaluation results — but not download or access original video files..""")

# ------------------ Data Availability Disclaimer ------------------
with st.sidebar.expander("🔐 Data Availability Disclaimer", expanded=False):
    st.markdown("""
    ### 🔐 Data Availability Disclaimer
    
    For security and compliance reasons, the actual video datasets are **not included** in the downloadable package.  
    
    However, we provide a **summary CSV file** listing the filenames, labels (real/fake), and source dataset (celeb or FaceForensics++) used in this project.
    
    🎓 If you are a data scientist or academic researcher, you may request access to the original datasets directly from the official sources:
    
    - [celeb-DF-v2](https://github.com/DigitalTrustLab/celeb-DF)
    - [FaceForensics++](https://github.com/ondyari/FaceForensics)
    
    You can then filter the videos using the filenames provided in this report to **recreate the age-diverse deepfake dataset** used here.
    
    _We appreciate your understanding and commitment to ethical data usage._
    """)

#------------------ Section for: UPLOAD REAL AND FAKE VIDEOS --------------
st.sidebar.subheader("📤 Upload Videos from celeb-DF")
real_videos_1 = st.sidebar.file_uploader("REAL Videos (celeb-DF)", type=["mp4", "avi"], accept_multiple_files=True, key="real_1")
fake_videos_1 = st.sidebar.file_uploader("FAKE Videos (celeb-DF)", type=["mp4", "avi"], accept_multiple_files=True, key="fake_1")

st.sidebar.markdown("---")

st.sidebar.subheader("📤 Upload Videos from FaceForensics++")
real_videos_2 = st.sidebar.file_uploader("REAL Videos (FaceForensics++)", type=["mp4", "avi"], accept_multiple_files=True, key="real_2")
fake_videos_2 = st.sidebar.file_uploader("FAKE Videos (FaceForensics++)", type=["mp4", "avi"], accept_multiple_files=True, key="fake_2")

#------------------ Section for: IMPORTING REAL AND FAKE VIDEOS --------------

if st.sidebar.button("📥 Import All Videos to Combined Folder"):
    log_action("Import Data", "SUCCESS", "Loaded from uploaded file.")
    clean_import_directory()
    
    base_output_dir = "all_data_videos"
    real_dir = os.path.join(base_output_dir, "real")
    fake_dir = os.path.join(base_output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Save celeb-DF
    if real_videos_1:
        for rv in real_videos_1:
            with open(os.path.join(real_dir, f"celeb_data_real_{rv.name}"), "wb") as f:
                f.write(rv.read())
    if fake_videos_1:
        for fv in fake_videos_1:
            with open(os.path.join(fake_dir, f"celeb_data_fake_{fv.name}"), "wb") as f:
                f.write(fv.read())

    # Save FaceForensics++
    if real_videos_2:
        for rv in real_videos_2:
            with open(os.path.join(real_dir, f"faceforensics_data_real_{rv.name}"), "wb") as f:
                f.write(rv.read())
    if fake_videos_2:
        for fv in fake_videos_2:
            with open(os.path.join(fake_dir, f"faceforensics_data_fake_{fv.name}"), "wb") as f:
                f.write(fv.read())

    st.sidebar.success("✅ Videos are successfully imported and saved to the folder `/all_data_videos`.")
    log_action("Import Data", "SUCCESS", "Loaded from uploaded file.")
    
#------------------ Section for: CLEANING LOW QUALITY VIDEO  --------------
st.markdown("---")
st.subheader("🧹🧼 Preprocess: Remove Low-Resolution Files")

output_video_dir = "all_data_videos"

# Info box
with st.expander("ℹ️ What This Does"):
    st.markdown("""
    - Scans all videos in `all_data_videos/real` and `all_data_videos/fake`.
    - Identifies low-quality or unreadable videos (e.g., corrupt, 0-byte, unreadable by OpenCV).
    - Deletes videos that fail the quality check.
    """)

# Initialize a flag in session state
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

# Display persistent success message
if st.session_state.clean_success:
    st.success("✅ Low-quality videos removed.")
#-------------------SECTION FOR: OPTIMIZED FRAME EXTRACTACTION FROM VIDEOS----------
st.markdown("---")
st.subheader("🖼️ Extract Frames")

# ℹ️ Use expander instead of modal
with st.expander("ℹ️ Frame Extraction Logic"):
    st.markdown("""
    ### 📘 How It Works:
    - Combines uploaded real and fake videos into a single folder.
    - Extracts frames from videos using OpenCV's `cv2.VideoCapture`, skipping every _N_ frames (user-defined).
    - Uses `ThreadPoolExecutor` to run extraction in parallel across CPU cores.
    - Saves frames as `.jpg` in the `all_data_frames/` directory.
    - Frames are labeled as real/fake based on video origin.
    - You can optionally overwrite previously extracted frames.
    """)

# 🛠️ Frame extraction settings
frame_rate = st.selectbox("Select frame extraction rate (every Nth frame)", options=[1, 2, 5, 10, 15, 20, 25, 30], index=3)
overwrite = st.checkbox("Overwrite existing frames if already extracted", value=False)

# ✅ Initialize frame output path in session state
if "frame_output_dir" not in st.session_state:
    st.session_state.frame_output_dir = None
# 👉 Move button to the right using columns
col1, col2, col3 = st.columns([2, 1, 1])

if col3.button("🚀 Run Frame Extraction"):
    combined_dir = st.session_state.get("combined_dir", "all_data_videos")  # fallback if not set

    if combined_dir and os.path.exists(combined_dir):
        st.session_state.frame_output_dir = extract_frames_from_combined_parallel(
            combined_dir,
            frame_rate=frame_rate,
            overwrite=overwrite,
            streamlit_mode=True,
            batch_mode=False,
            num_workers=None
        )
        st.session_state.frame_extraction_success = True
        log_action("Frame Extraction", "SUCCESS", "Data frames were successfully extracted from video")
    else:
        st.error("❌ Combined video directory not found.")
        st.session_state.frame_extraction_success = False
        log_action("Frame Extraction", "FAIL", "Data frames extraction was not successful")

    if st.session_state.frame_extraction_success:
        st.success(f"✅ Frames successfully extracted to: `{st.session_state.frame_output_dir}`")

#------------------ Section for: DISPLAY SAMPLE FRAMES--------------
st.markdown("---")
st.subheader("🔍 Preview Sample Frames")

# ℹ️ Expander with logic explanation
with st.expander("ℹ️ What This Does"):
    st.markdown("""
    - Randomly selects **up to 9 real** and **9 fake** videos from your dataset.
    - Extracts **one frame per video** (first valid frame).
    - Displays frames side-by-side in a 3×3 grid for each class.
    - Does **not save** extracted frames — only used for previewing visual quality.
    """)

# 📂 Define your folder paths (used earlier when importing videos)
real_video_dir = os.path.join("all_data_videos", "real")
fake_video_dir = os.path.join("all_data_videos", "fake")

col1, col2, col3 = st.columns([2, 1, 1])
# Init session_state
if "preview_real_grid" not in st.session_state:
    st.session_state.preview_real_grid = []
if "preview_fake_grid" not in st.session_state:
    st.session_state.preview_fake_grid = []
if "preview_done" not in st.session_state:
    st.session_state.preview_done = False

# Run preview and store frames
if col3.button("🖼️ Preview Frames"):
    if os.path.exists(real_video_dir) and os.path.exists(fake_video_dir):
        real_grid, fake_grid = preview_sample_frames(real_video_dir, fake_video_dir, return_images=True)
        st.session_state.preview_real_grid = real_grid
        st.session_state.preview_fake_grid = fake_grid
        st.session_state.preview_done = True
        log_action("Previe Frame", "SUCCESS", "Data frames were successfully displayed")
    else:
        st.error("❌ Please upload and save real/fake videos first.")
        st.session_state.preview_done = False
        log_action("Previe Frame", "FAIL", "Fail to show Data frames")

# Display persisted grid using your original layout
if st.session_state.preview_done:
    display_grid_pair(st.session_state.preview_real_grid, st.session_state.preview_fake_grid)

#------------------ Section for: AGE ANNOTATION FOR EXISTING VIDEOS --------------
st.markdown("---")
st.subheader("🧓 Annotate and Save Age Groups")

# ℹ️ Explanation using expander
with st.expander("ℹ️ What This Does"):
    st.markdown("""
    - Reads the **first frame** of each video in the `all_data_videos/` directory.
    - Uses `DeepFace.analyze()` to estimate age from the face in the frame.
    - Converts the age into one of these age groups: `0-10`, `10-19`, `19-35`, `36-50`, or `51+`.
    - Processes videos in parallel using `ThreadPoolExecutor` for faster annotation.
    - Handles errors by assigning a **random age group** if detection fails.
    - Saves a CSV file `annotations.csv` with:
        - `filename`  
        - `path`  
        - `label` (`real` or `fake`)  
        - `age_group`
    """)
# 📂 Directory containing videos
video_annotation_dir = "all_data_videos"
frame_dir = "all_data_frames"

# ✅ Initialize session state
if "annotations_df" not in st.session_state:
    st.session_state.annotations_df = None
if "age_annotation_done" not in st.session_state:
    st.session_state.age_annotation_done = False

# 🔘 Move button to the right
col1, col2, col3 = st.columns([2, 1, 1])
if col3.button("📋 Label Age Group"):
    if os.path.exists(video_annotation_dir) and os.path.exists(frame_dir):
        progress_bar = st.progress(0)
        df = save_age_annotations_parallel(
            video_dir=video_annotation_dir,
            output_csv="annotations.csv",
            batch_mode=True,
            num_workers=None,
            streamlit_progress=progress_bar
        )
        st.session_state.annotations_df = df
        st.session_state.age_annotation_done = True
        log_action("Age Annotation", "SUCCESS", "Age annotation success and metadata saved")

        generate_frame_level_annotations(mode="annotated")
    else:
        st.error("❌ Folder not found: `all_data_videos`. Please upload and combine videos first.")
        st.session_state.age_annotation_done = False
        log_action("Age Annotation", "FAIL", "Age annotation could not complete")

# ✅ Display message and sample preview on rerun
if st.session_state.age_annotation_done and st.session_state.annotations_df is not None:
    st.success("✅ Age annotation complete and metadata saved to `annotations.csv`.")
    st.success("📝 Saved backup: frame_level_annotations_original.csv")


#------------------ Section for: VISUALIZING THE AGE DISTIBUTION --------------

st.markdown("---")
st.subheader("📊 Age & Label Distribution by Dataset")

##---------Table----------------


if "age_dist_table" not in st.session_state:
    st.session_state.age_dist_table = None
if "show_age_dist_table" not in st.session_state:
    st.session_state.show_age_dist_table = False

# Show Distribution Table Button
if st.button("📊 Show Distribution Table"):
    pivot = get_age_label_source_table()
    if pivot is not None:
        st.session_state.age_dist_table = pivot
        st.session_state.show_age_dist_table = True
        log_action("Age distribution table", "SUCCESS", "Distribution table successfully loaded")
    else:
        st.error("❌ `annotations.csv` not found. Please run age annotation first.")
        st.session_state.show_age_dist_table = False
        log_action("Age distribution table", "FAIL", "Could not load Distribution table")

# Persistent Display of Table
if st.session_state.show_age_dist_table and st.session_state.age_dist_table is not None:
    st.dataframe(st.session_state.age_dist_table)


        
##---------------Piechart-----------

if "pie_chart_figures" not in st.session_state:
    st.session_state.pie_chart_figures = []
if "show_pie_charts" not in st.session_state:
    st.session_state.show_pie_charts = False

# Trigger pie chart generation
if st.button("🥧 Show Age Group Pie Charts"):
    st.session_state.pie_chart_figures = show_age_distribution_pie_charts(return_figures=True)
    st.session_state.show_pie_charts = True

# Display persisted pie charts
if st.session_state.show_pie_charts and st.session_state.pie_chart_figures:
    st.markdown("### 🥧 Age Group Distribution by Source")
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    for source, fig in st.session_state.pie_chart_figures:
        if source == "celeb":
            with col2:
                st.markdown("#### 🟢 celeb")
                st.pyplot(fig)
                log_action("Source Distibution", "SUCCESS", "Distribution pie chart successfully loaded")
        elif source == "FaceForensics++":
            with col3:
                st.markdown("#### 🔵 FaceForensics++")
                st.pyplot(fig)
                log_action("Source Distibution", "FAIL", "Distribution pie chart could not load")

    
#------------------ Section for: AGE BALANCE DATASET --------------

st.markdown("---")
st.subheader("⚙️ Preprocess & Balance Dataset")
with st.expander("ℹ️ What This Does"):
    st.markdown("""
    This module balances the dataset across **age groups** using either:
    
    - **Undersampling**: Reduces overrepresented classes
    - **Oversampling**: Replicates underrepresented classes
    
    It then:
    - Saves a new metadata file: `balanced_annotations.csv`
    - Generates updated `frame_level_annotations.csv`
    - Copies balanced video frames into a new dataset directory
    """)
balance_method = st.radio("Select balancing method:", ["undersample", "oversample"], horizontal=True)

# Initialize session_state
if "balance_result" not in st.session_state:
    st.session_state.balance_result = None
if "balance_done" not in st.session_state:
    st.session_state.balance_done = False

if st.button("⚖️ Balance Dataset"):
    from balance_data import balance_and_export_dataset
    result = balance_and_export_dataset(method=balance_method)
    st.session_state.balance_result = result
    st.session_state.balance_done = True if result["status"] == "success" else False

# Display if balance was successful
if st.session_state.balance_done and st.session_state.balance_result:
    result = st.session_state.balance_result

    st.markdown("🔍 Age-Distribution Before Balancing")
    st.dataframe(result["pre_distribution"])

    st.markdown("🔍 Age-Distribution After Balancing")
    st.dataframe(result["post_distribution"])

    if result["frame_annotated"]:
        st.success("🧾 Frame-level annotations saved to `frame_level_annotations.csv`.")

    st.success(f"📦 Export complete: {result['copied']} videos copied to `{result['export_path']}/age_diverse_videos/`.")
    st.info("📝 Metadata saved as `balanced_annotations.csv` in `final_output/`.")
    log_action("Data Balance", "SUCCESS", "Successfuly balanced the data and stored in database")

    if result["missing"]:
        st.warning(f"⚠️ {len(result['missing'])} videos listed in CSV were not found on disk.")
        st.write(result["missing"])

elif st.session_state.balance_result and st.session_state.balance_result["status"] == "error":
    for msg in st.session_state.balance_result["messages"]:
        st.error(msg)
        log_action("Data Balance", "FAIL", "Fail to balance the dataset")

#------------------ Section for: SYNTHETIC DATA --------------
# st.markdown("---")
# st.subheader("🌀 Generate Synthetic Deepfakes with SimSwap")

# with st.expander("ℹ️ What This Does", expanded=False):
#     st.markdown("""
#     This module creates synthetic deepfake videos using the **SimSwap** model.
    
#     - Upload one or more **source face images** (e.g., of different age groups).
#     - Upload one or more **target videos** (e.g., real videos from your dataset).
#     - When click **Run SimSwap**, all files are saved into my local **Google Drive** sync folder:
#       `G:/My Drive/SyntheticData/simswap_batch_input`
#     - THen the code is run in the SimSwap Colab notebook, which reads from this folder and generates face-swapped videos.
#     - The output includes:
#       - Swapped fake videos (for use in your dataset)
#       - A `swap_metadata.csv` file that logs source–target pairs
#     """)
    
# source_faces = st.file_uploader(
#     "Upload Source Face Images", type=["jpg", "jpeg", "png"],
#     accept_multiple_files=True, key="faces"
# )

# target_videos = st.file_uploader(
#     "Upload Target Videos", type=["mp4", "avi"],
#     accept_multiple_files=True, key="targets"
# )

# if st.button("🚀 Run SimSwap Face Swapping"):
#     from synthetic_data import run_simswap_batch

#     success = run_simswap_batch(source_faces, target_videos)

#     if success:
#         st.success("✅ Exported to Google Drive! You can now run SimSwap from Colab.")
#     else:
#         st.error("❌ Export failed. Check file paths and retry.")

st.markdown("---")
##-----------Balanced dataset Age Distribution--------------
st.subheader("📈 Visualize Age-Balanced Dataset")
# Initialize session state
if "age_dist_figure" not in st.session_state:
    st.session_state.age_dist_figure = None
if "show_age_dist_plot" not in st.session_state:
    st.session_state.show_age_dist_plot = False

# 📊 Visualize Button
if st.button("📊 View Age Distribution"):
    try:
        df = pd.read_csv("balanced_annotations.csv")
        fig = visualize_age_distribution(df)

        # Store in session state
        st.session_state.age_dist_figure = fig
        st.session_state.show_age_dist_plot = True
        log_action("Visualiza Balance Dataset", "SUCCESS", "Balance the dataset histogram plotted")

    except FileNotFoundError:
        st.error("balanced_annotations.csv not found. Please process balance dataset first.")
        st.session_state.show_age_dist_plot = False
        log_action("Visualiza Balance Dataset", "FAIL", "Fail to visualize balance the dataset")

# ✅ Persisted Plot Display
if st.session_state.show_age_dist_plot and st.session_state.age_dist_figure is not None:
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.pyplot(st.session_state.age_dist_figure)



st.markdown("---")
#------------------ Section for: MODEL TRAINING and Evaluation --------------

st.subheader("Deepfake Detection Model")

##------------Train and Test Split-----------
# -------------------------
# 🔄 Train-Test Split
# -------------------------
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

# -------------------------
# 🏋️ Train Models
# -------------------------
st.markdown("### 🧠 Model Training")
with st.expander("ℹ️ What This Does"):
    st.markdown("""
    This section allows to **train one or more deepfake detection models** using the balanced dataset.

    You can select from:
    - **XceptionNet**
    - **EfficientNet**
    - **LipForensics**

    These models are trained on the **70% training split** generated in the previous step, which is balanced by **age group** to ensure fairness and reduce bias in predictions.
    """)

# ✅ Initialize session state
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "training_done" not in st.session_state:
    st.session_state.training_done = False

# ✅ Model selection checkboxes (don’t clear on rerun)
col1, col2, col3 = st.columns(3)
with col1:
    if st.checkbox("XceptionNet", value="XceptionNet" in st.session_state.selected_models):
        if "XceptionNet" not in st.session_state.selected_models:
            st.session_state.selected_models.append("XceptionNet")
    else:
        if "XceptionNet" in st.session_state.selected_models:
            st.session_state.selected_models.remove("XceptionNet")
with col2:
    if st.checkbox("EfficientNet", value="EfficientNet" in st.session_state.selected_models):
        if "EfficientNet" not in st.session_state.selected_models:
            st.session_state.selected_models.append("EfficientNet")
    else:
        if "EfficientNet" in st.session_state.selected_models:
            st.session_state.selected_models.remove("EfficientNet")
with col3:
    if st.checkbox("LipForensics", value="LipForensics" in st.session_state.selected_models):
        if "LipForensics" not in st.session_state.selected_models:
            st.session_state.selected_models.append("LipForensics")
    else:
        if "LipForensics" in st.session_state.selected_models:
            st.session_state.selected_models.remove("LipForensics")

# ✅ Trigger training
if st.button("Train Selected Models"):
    if not st.session_state.selected_models:
        st.warning("⚠️ Please select at least one model.")
        st.session_state.training_done = False
    else:
        with st.spinner("Training in progress..."):
            from model_trainer import train_models
            st.session_state.trained_models = train_models(
                st.session_state.selected_models,
                train_csv="train_split.csv",
                streamlit_mode=True
            )
        st.session_state.training_done = True

if st.session_state.training_done:
    trained_list = ", ".join(st.session_state.selected_models)
    st.success(f"✅ Training complete for: {trained_list}")
    log_action("Model Training", "SUCCESS", "Deepfake models are successfully trained using balanced dataset")

st.markdown("---")   
# -------------------------
# 📊 Evaluate Trained Models
# -------------------------
st.markdown("### 📊 Evaluate Models on Balanced, Colab, FF++")

# ✅ Initialize session state
if "eval_df" not in st.session_state:
    st.session_state.eval_df = None
if "eval_done" not in st.session_state:
    st.session_state.eval_done = False

if st.button("Run Evaluation", key="eval_button"):
    if not st.session_state.selected_models:
        st.warning("⚠️ Please train models first.")
        st.session_state.eval_done = False
    else:
        with st.spinner("Evaluating models..."):
            results = evaluate_on_all_sets(
                selected_models=st.session_state.selected_models,
                streamlit_mode=True
            )
        st.success("✅ Evaluation complete.")
        log_action("Model Evaluation", "SUCCESS", "Successful model evalutaion")
        st.session_state.eval_df = flatten_results_grouped(results)
        st.session_state.eval_done = True
        
        

        # Save to disk
        os.makedirs("final_output", exist_ok=True)
        eval_csv_path = "final_output/evaluation_results.csv"
        st.session_state.eval_df.to_csv(eval_csv_path, index=False)

# 
if st.session_state.eval_done and st.session_state.eval_df is not None:
    st.dataframe(st.session_state.eval_df, use_container_width=True)
    log_action("Model Evaluation", "SUCCESS", "Model evalutaion results successfully plotted")
    st.download_button(
        label="⬇️ Download Evaluation Summary",
        data=st.session_state.eval_df.to_csv(index=False),
        file_name="evaluation_results.csv",
        mime="text/csv"
    )
st.markdown("---")   

####-------------------------Age Specific evaluation------------------
st.markdown("### 📊 Age-Specific Evaluation by Dataset and Age Group")

# ✅ Initialize session state
if "age_eval_df" not in st.session_state:
    st.session_state.age_eval_df = None
if "age_eval_done" not in st.session_state:
    st.session_state.age_eval_done = False

# Run evaluation
if st.button("Run Age Group Evaluation", key="age_eval_button"):
    if not st.session_state.selected_models:
        st.warning("⚠️ Please train models first.")
        st.session_state.age_eval_done = False
    else:
        with st.spinner("Running age-specific evaluation..."):
            age_results = evaluate_on_all_sets_agewise(
                selected_models=st.session_state.selected_models,
                streamlit_mode=True
            )

        st.success("✅ Age-specific evaluation complete.")
        log_action("Age-specific evaluation", "SUCCESS", "Age-specific mModel evalutaion results successfully plotted")
        df_age_eval = flatten_age_specific_results(age_results)

        # Store results
        st.session_state.age_eval_df = df_age_eval
        st.session_state.age_eval_done = True

        # Save to disk
        os.makedirs("final_output", exist_ok=True)
        age_eval_csv_path = "final_output/age_specific_evaluation.csv"
        df_age_eval.to_csv(age_eval_csv_path, index=True)


if st.session_state.age_eval_done and st.session_state.age_eval_df is not None:
    st.dataframe(st.session_state.age_eval_df, use_container_width=True)

    st.download_button(
        label="⬇️ Download Age-Specific Evaluation",
        data=st.session_state.age_eval_df.to_csv(index=True),
        file_name="age_specific_evaluation.csv",
        mime="text/csv"
    )

st.markdown("---")
# -------------------  Section: Train with Different Training Sources-------------------
st.markdown("## 🔄 Deepfake Detection with Original Data")

st.markdown("#### ✅ Select Models to Train")
st.markdown("### 🏋️ Source-Specific Model Training")

# ✅ Initialize session state variables
if "selected_models" not in st.session_state:
    st.session_state.selected_models = []
if "source_training_done" not in st.session_state:
    st.session_state.source_training_done = False
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "test_df" not in st.session_state:
    st.session_state.test_df = None

# ✅ 3-column checkbox layout
col1, col2, col3 = st.columns(3)

with col1:
    if st.checkbox("XceptionNet", value="XceptionNet" in st.session_state.selected_models, key="celeb_train_xcep"):
        if "XceptionNet" not in st.session_state.selected_models:
            st.session_state.selected_models.append("XceptionNet")
    else:
        if "XceptionNet" in st.session_state.selected_models:
            st.session_state.selected_models.remove("XceptionNet")

with col2:
    if st.checkbox("EfficientNet", value="EfficientNet" in st.session_state.selected_models, key="celeb_train_eff"):
        if "EfficientNet" not in st.session_state.selected_models:
            st.session_state.selected_models.append("EfficientNet")
    else:
        if "EfficientNet" in st.session_state.selected_models:
            st.session_state.selected_models.remove("EfficientNet")

with col3:
    if st.checkbox("LipForensics", value="LipForensics" in st.session_state.selected_models, key="celeb_train_lip"):
        if "LipForensics" not in st.session_state.selected_models:
            st.session_state.selected_models.append("LipForensics")
    else:
        if "LipForensics" in st.session_state.selected_models:
            st.session_state.selected_models.remove("LipForensics")

# ✅ Source selector
train_source = st.radio("Select training dataset source:", ["celeb", "FaceForensics++"], horizontal=True, key="train_source_radio")

# ✅ Train Button
if st.button("🏋️ Train Models", key="train_src_button"):
    if not st.session_state.selected_models:
        st.warning("⚠️ Please select at least one model.")
        st.session_state.source_training_done = False
    else:
        from model_trainer import train_models_on_source
        source_name = train_source.lower()

        with st.spinner(f"Training models on {train_source} dataset..."):
            trained_models, test_df = train_models_on_source(
                source_name=source_name,
                metadata_csv="frame_level_annotations_original.csv",
                selected_models=st.session_state.selected_models,
                streamlit_mode=True
            )

        st.session_state.trained_models = trained_models
        st.session_state.test_df = test_df
        st.session_state.source_training_done = True

# ✅ Persistent success message with model list
if st.session_state.source_training_done:
    trained_list = ", ".join(st.session_state.selected_models)
    st.success(f"✅ Model training complete for {train_source}: {trained_list}")

st.markdown("---")
# -------------------  Evaluate with Different Training Sources-------------------
# Step 3: Evaluate Button
st.markdown("### 🧪 Evaluate Original Source Performance")

# ✅ Initialize session state
if "cross_eval_df" not in st.session_state:
    st.session_state.cross_eval_df = None
if "cross_eval_done" not in st.session_state:
    st.session_state.cross_eval_done = False

# 🔘 Evaluation trigger
if st.button("🔍 Evaluate Across All Test Sets", key="cross_eval_button"):
    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.warning("⚠️ Please train models first.")
        st.session_state.cross_eval_done = False
    else:
        from evaluation import evaluate_on_all_sets_for_trained_models, flatten_results_grouped

        with st.spinner("🔍 Evaluating models on Balanced, celeb, FF++ test sets..."):
            cross_results = evaluate_on_all_sets_for_trained_models(
                st.session_state.trained_models,
                streamlit_mode=True
            )

        df_cross_eval = flatten_results_grouped(cross_results)

        st.session_state.cross_eval_df = df_cross_eval
        st.session_state.cross_eval_done = True

        # Save for download
        os.makedirs("final_output", exist_ok=True)
        source_str = st.session_state.get("train_source_radio", "unknown").lower()
        cross_csv_path = f"final_output/{source_str}_cross_eval.csv"
        df_cross_eval.to_csv(cross_csv_path, index=False)

# ✅ Persist results and download
if st.session_state.cross_eval_done and st.session_state.cross_eval_df is not None:
    st.success("✅ Cross-evaluation complete!")
    st.dataframe(st.session_state.cross_eval_df, use_container_width=True)

    st.download_button(
        label="⬇️ Download Cross-Evaluation Summary",
        data=st.session_state.cross_eval_df.to_csv(index=False),
        file_name="cross_evaluation_results.csv",
        mime="text/csv"
    )


# ------------------- 🎯 Grad-CAM Age-Specific Visualization -------------------
st.markdown("---")
st.subheader("🎯 Age-Specific Deepfake Explanation (Grad-CAM)")

# 🔍 Dynamically find the last convolutional layer
def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    return None

# 🖼️ Grid Display Function
def display_images_in_grid(image_list, captions, title, columns=4):
    st.markdown(f"#### {title}")
    for i in range(0, len(image_list), columns):
        cols = st.columns(columns)
        for j in range(columns):
            idx = i + j
            if idx < len(image_list):
                cols[j].image(image_list[idx], caption=captions[idx], use_container_width=True)

# ✅ Initialize session state
for key in ["gradcam_real_images", "gradcam_fake_images", "gradcam_real_captions", "gradcam_fake_captions", "gradcam_done"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ℹ️ Description Section
with st.expander("ℹ️ How It Works", expanded=False):
    st.markdown("""
    This tool highlights the image regions that the selected deepfake model relies on for classification.
    It selects **4 real** and **4 fake** samples from different videos in the balanced dataset and displays Grad-CAM heatmaps.
    """)

# Model Selection Dropdown
selected_model = st.selectbox("🤖 Choose a Model", ["XceptionNet", "EfficientNet", "LipForensics"], key="gradcam_model")

# 🔘 Grad-CAM Run Button
if st.button("🎯 Run Grad-CAM"):
    if selected_model:
        with st.spinner("🔍 Processing 8 unique frames..."):
            if not os.path.exists("frame_level_annotations.csv"):
                st.error("⚠️ `frame_level_annotations.csv` not found.")
            else:
                df = pd.read_csv("frame_level_annotations.csv")
                df["video_id"] = df["frame"].apply(lambda x: os.path.splitext(x)[0])

                real_videos = df[df["label"] == "real"].drop_duplicates("video_id")
                fake_videos = df[df["label"] == "fake"].drop_duplicates("video_id")

                real_sample = real_videos.sample(n=min(4, len(real_videos)), random_state=1)
                fake_sample = fake_videos.sample(n=min(4, len(fake_videos)), random_state=2)

                combined_df = pd.concat([real_sample, fake_sample])
                frame_paths = combined_df["path"].tolist()
                labels = combined_df["label"].tolist()

                model = get_model(selected_model)
                target_layer = get_last_conv_layer(model)

                if selected_model == "XceptionNet":
                    target_layer = getattr(model, "conv2", target_layer)
                elif selected_model == "EfficientNet":
                    target_layer = getattr(model, "conv_head", target_layer)
                elif selected_model == "LipForensics":
                    target_layer = getattr(model, "features", [])[ -1 ] if hasattr(model, "features") else target_layer

                if target_layer is None:
                    st.error("❌ Could not detect a valid convolutional layer.")
                else:
                    real_images, fake_images = [], []
                    real_captions, fake_captions = [], []

                    for path, label in zip(frame_paths, labels):
                        if not os.path.exists(path): continue
                        try:
                            cam_result = apply_gradcam(model, path, target_layer)
                            if label == "real":
                                real_images.append(cam_result)
                                real_captions.append(os.path.basename(path))
                            else:
                                fake_images.append(cam_result)
                                fake_captions.append(os.path.basename(path))
                        except Exception as e:
                            st.warning(f"⚠️ Error on {path}: {e}")

                    # ✅ Store in session state
                    st.session_state.gradcam_real_images = real_images
                    st.session_state.gradcam_fake_images = fake_images
                    st.session_state.gradcam_real_captions = real_captions
                    st.session_state.gradcam_fake_captions = fake_captions
                    st.session_state.gradcam_done = True
    else:
        st.warning("Please select a model to proceed.")

# ✅ Display persistent Grad-CAM results
if st.session_state.gradcam_done:
    st.markdown("### 🎨 Grad-CAM Results")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        display_images_in_grid(st.session_state.gradcam_real_images, st.session_state.gradcam_real_captions, title="🟢 Real Samples")
        display_images_in_grid(st.session_state.gradcam_fake_images, st.session_state.gradcam_fake_captions, title="🔴 Fake Samples")

st.markdown("---")   
#------------------ Section for: Project Report PDF Download Export --------------
# Section: Export Results
st.subheader("📄 Download Project Summary Report")

# ✅ Initialize session state
if "pdf_report_path" not in st.session_state:
    st.session_state.pdf_report_path = None
if "pdf_report_done" not in st.session_state:
    st.session_state.pdf_report_done = False

# 📝 Generate PDF Report
if st.button("📝 Generate PDF Report"):
    with st.spinner("Generating PDF report..."):
        dataset_summary = {
            "celeb_real": len([f for f in os.listdir("all_data_videos/real") if "celeb" in f.lower()]),
            "celeb_fake": len([f for f in os.listdir("all_data_videos/fake") if "celeb" in f.lower()]),
            "ffpp_real": len([f for f in os.listdir("all_data_videos/real") if "faceforensics" in f.lower()]),
            "ffpp_fake": len([f for f in os.listdir("all_data_videos/fake") if "faceforensics" in f.lower()])
        }

        cleaning_notes = "- Low-resolution or corrupted files were removed during preprocessing."

        frame_count_summary = len(os.listdir("all_data_frames")) if os.path.exists("all_data_frames") else 0

        sample_frame_paths = []
        for root, _, files in os.walk("all_data_frames"):
            for f in files:
                if f.endswith(".jpg"):
                    sample_frame_paths.append(os.path.join(root, f))
        sample_frame_paths = sample_frame_paths[:6]

        age_annotation_summary = {}
        if os.path.exists("annotations.csv"):
            age_df = pd.read_csv("annotations.csv")
            age_annotation_summary = age_df["age_group"].value_counts().to_dict()

        balance_summary = {}
        if os.path.exists("balanced_metadata.csv"):
            balanced_df = pd.read_csv("balanced_metadata.csv")
            balance_summary = balanced_df["age_group"].value_counts().to_dict()

        eval_balanced = pd.read_csv("final_output/evaluation_results.csv") if os.path.exists("final_output/evaluation_results.csv") else None
        eval_agewise = pd.read_csv("final_output/age_specific_evaluation.csv") if os.path.exists("final_output/age_specific_evaluation.csv") else None
        eval_celeb = pd.read_csv("final_output/celeb_cross_eval.csv") if os.path.exists("final_output/celeb_cross_eval.csv") else None
        eval_ffpp = pd.read_csv("final_output/faceforensics_cross_eval.csv") if os.path.exists("final_output/faceforensics_cross_eval.csv") else None

        gradcam_paths = []
        for f in os.listdir("temp_uploads") if os.path.exists("temp_uploads") else []:
            if f.endswith(".jpg") or f.endswith(".png"):
                gradcam_paths.append(os.path.join("temp_uploads", f))

        # 📄 Call PDF generator
        report_path = create_full_pdf_report(
            output_path="final_output/final_report.pdf",
            dataset_summary=dataset_summary,
            cleaning_notes=cleaning_notes,
            frame_count_summary=frame_count_summary,
            sample_frame_paths=sample_frame_paths,
            age_annotation_summary=age_annotation_summary,
            balance_summary=balance_summary,
            eval_balanced=eval_balanced,
            eval_agewise=eval_agewise,
            eval_celeb=eval_celeb,
            eval_ffpp=eval_ffpp,
            gradcam_paths=gradcam_paths
        )

        st.session_state.pdf_report_path = report_path
        st.session_state.pdf_report_done = True

# ✅ Persist download button if report is ready
if st.session_state.pdf_report_done and st.session_state.pdf_report_path:
    with open(st.session_state.pdf_report_path, "rb") as f:
        st.success("✅ PDF report generated!")
        st.download_button(
            label="⬇️ Download Project Summary Report",
            data=f,
            file_name="Deepfake_Project_Report.pdf",
            mime="application/pdf"
        )


#------------------ Section for: Age-Diverse Deepfake Dataset Export --------------

st.markdown("---")

st.subheader("📦 Download Age-Diverse Data List & Results")

if st.button("📥 Download Details"):
    output_dir = "final_output"
    zip_buffer = BytesIO()

    # Step 1: Create video index summary
    video_summary = []
    for label in ["real", "fake"]:
        folder = os.path.join("all_data_videos", label)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith((".mp4", ".avi")):
                    source = "celeb" if "celeb" in filename.lower() else "faceforensics"
                    
                    # Truncate prefix from filename (e.g., "celeb_data_real_video123.mp4" -> "video123.mp4")
                    truncated_name = filename
                    for prefix in ["celeb_data_real_", "celeb_data_fake_", "faceforensics_data_real_", "faceforensics_data_fake_"]:
                        if filename.startswith(prefix):
                            truncated_name = filename.replace(prefix, "", 1)
                            break
                    
                    video_summary.append({
                        "filename": truncated_name,
                        "label": label,
                        "source": source
                    })

    # Save summary CSV
    summary_df = pd.DataFrame(video_summary)
    summary_csv_path = os.path.join(output_dir, "video_index.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # Step 2: Prepare ZIP archive with filtered content
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(output_dir):
            for filename in filenames:
                if filename.endswith((".csv", ".pdf")):
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, start=output_dir)
                    zipf.write(file_path, arcname=arcname)

    zip_buffer.seek(0)

    st.download_button(
        label="⬇️ Download ZIP Archive",
        data=zip_buffer,
        file_name="age_diverse_dataset_outputs.zip",
        mime="application/zip"
    )
st.markdown("""
### 🔐 Data Availability Disclaimer

For security and compliance reasons, the actual video datasets are **not included** in the downloadable package.  

However, we provide a **summary CSV file** listing the filenames, labels (real/fake), and source dataset (celeb or FaceForensics++) used in this project.

🎓 If you are a data scientist or academic researcher, you may request access to the original datasets directly from the official sources:

- [celeb-DF-v2](https://github.com/DigitalTrustLab/celeb-DF)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

You can then filter the videos using the filenames provided in this report to **recreate the age-diverse deepfake dataset** used here.

_We appreciate your understanding and commitment to ethical data usage._
""")
### ------------------- Thankyou section -------------------------
st.markdown("---")
st.markdown("### 🙏 Thank You!")
st.markdown("""
I appreciate your time exploring the **Age-Diverse Deepfake Dataset Builder**!  
This tool was designed to support research and experimentation in fairness-aware deepfake detection.

📫 **Contact**: [unishajoshi@email.com](mailto:unishajoshi@email.com)  
  
🏫 **Institution**: Department of Data Science, Grand Canyon University

Feel free to reach out for improvements, collaboration, or to share your findings!
""")

##/* 💻 **GitHub**: [github.com/unishajoshi/deepfake-dataset-builder](https://github.com/unishajoshi/deepfake-dataset-builder)*/
