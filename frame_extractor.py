#-------------------SECTION FOR: OPTIMIZED FRAME EXTRACTACTION FROM VIDEOS----------

import os
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from frame_filter import is_valid_frame 

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
    processed = 0
    filtered_out = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        if not success:
            break

        processed += 1
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

        # ✅ Check frame quality before saving
        if is_valid_frame(frame):
            cv2.imwrite(frame_path, frame)
            saved += 1 
        else:
            filtered_out += 1

        count += frame_rate

    cap.release()
    #print(f"✅ Frame extraction & filtering complete: {video_name} → {saved} high-quality frames saved")
    print(f"✅ Frame extraction & filtering complete: {video_name}")
    print(f"   Total frames processed: {processed}")
    print(f"   Saved: {saved}  |  Filtered out: {filtered_out}")
    return f"{video_name}: {saved} of {processed} frames saved"
   # return f"{video_name}: {saved} frames saved"

def _process_single_video_1(video_path, output_dir, frame_rate=10, overwrite=False):
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

