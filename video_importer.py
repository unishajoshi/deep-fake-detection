import os
import shutil
import random
import cv2
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

#------------------ Section for: COMBINING REAL AND FAKE VIDEOS --------------

def combine_videos(real_dir, fake_dir, combined_dir="all_data_videos"):
    os.makedirs(combined_dir, exist_ok=True)

    def copy_videos(src_dir, label):
        for filename in os.listdir(src_dir):
            if filename.endswith(('.mp4', '.avi')):
                src_path = os.path.join(src_dir, filename)
                dst_filename = f"{label}_{filename}"
                dst_path = os.path.join(combined_dir, dst_filename)
                shutil.copy(src_path, dst_path)

    copy_videos(real_dir, "real")
    copy_videos(fake_dir, "fake")
    return combined_dir

#------------------ Section for: DISPLAY SAMPLE FRAMES--------------

def display_grid_pair(real_frames, fake_frames):

    col1, col2 = st.columns(2)

    def show_grid(frames, label, container,border_color):
        with container:
            container.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; margin-bottom: 20px;">
                    <h4 style='text-align: center;'>{label}</h4>
                """,
                unsafe_allow_html=True
            )
            rows, cols = 3, 3
            square_size = 150
            for i in range(rows):
                columns = st.columns(cols)
                for j in range(cols):
                    index = i * cols + j
                    if index < len(frames):
                        img, _ = frames[index]
                        img_resized = cv2.resize(img, (square_size, square_size))
                        columns[j].image(img_resized, channels="BGR", width=square_size)
                    else:
                        columns[j].empty()

    show_grid(real_frames, "🟢 Real", col1, "#4CAF50")
    show_grid(fake_frames, "🔴 Fake", col2, "#F44336")

def preview_sample_frames(real_dir, fake_dir, return_images=False, frame_output_dir="sample_preview_frames", frame_rate=10, count=9):
    os.makedirs(frame_output_dir, exist_ok=True)

    def get_sample_frames(video_dir, label):
        sample_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        sample_videos = random.sample(sample_videos, min(count, len(sample_videos)))
        preview_frames = []

        for video in sample_videos:
            video_path = os.path.join(video_dir, video)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_rate)
            success, frame = cap.read()
            cap.release()
            if success:
                preview_frames.append((frame, f"{label}: {video}"))  # keep frame as BGR np.array

        return preview_frames

    real_frames = get_sample_frames(real_dir, "Real")
    fake_frames = get_sample_frames(fake_dir, "Fake")

    if return_images:
        return real_frames, fake_frames  # ✅ raw BGR frames with captions
    else:
        display_grid_pair(real_frames, fake_frames)


#-------------------SECTION FOR: OPTIMIZED FRAME EXTRACTACTION FROM VIDEOS----------

# Check once if CUDA is available
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print("📸 OpenCV CUDA Devices:", cv2.cuda.getCudaEnabledDeviceCount())
print("🚀 PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 PyTorch device:", torch.cuda.get_device_name(0))

def _process_single_video(video_path, output_dir, frame_rate=10, overwrite=False):
    label = "real" if "real" in video_path else "fake"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        if not success:
            break

        frame_name = f"{label}_{video_name}_frame{count}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        if not overwrite and os.path.exists(frame_path):
            count += frame_rate
            continue

        if cuda_available:
            try:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(frame)
                frame = gpu_mat.download()
            except Exception as e:
                print(f"⚠️ CUDA fallback for {frame_name}: {e}")

        cv2.imwrite(frame_path, frame)
        saved += 1
        count += frame_rate

    cap.release()
    print(f" Frame extraction successful: {video_name}")
    return f"{video_name}: {saved} frames saved"


def extract_frames_from_combined_parallel(
    combined_dir, frame_output="all_data_frames", frame_rate=10,
    overwrite=False, streamlit_mode=False, batch_mode=False, num_workers=4
):
    if streamlit_mode and not batch_mode:
        import streamlit as st
    else:
        st = None

    os.makedirs(frame_output, exist_ok=True)
    # Recursively collect all videos
    video_files = []
    for root, _, files in os.walk(combined_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    results = []
    if streamlit_mode and not batch_mode:
        st.write(f"📼 Processing {len(video_files)} videos with {num_workers} threads...")
        progress_bar = st.progress(0)
    else:
        progress_bar = None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_video, path, frame_output, frame_rate, overwrite): path
            for path in video_files
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if progress_bar:
                progress_bar.progress((i + 1) / len(video_files))

    if progress_bar:
        progress_bar.empty()

    return frame_output
