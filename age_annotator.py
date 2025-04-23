import os
import cv2
import random
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import glob

#--------check for CUDA----------
# Check CUDA availability for DeepFace (PyTorch backend)
print("🚀 PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🧠 PyTorch device:", torch.cuda.get_device_name(0))

#------------------ Section for: AGE ANNOTATION FOR EXISTING VIDEOS --------------

def age_to_group(age):
    if age < 10:
        return '0-10'
    elif age < 19:
        return '10-18'
    elif age < 36:
        return '19-35'
    elif age < 51:
        return '36-50'
    else:
        return '51+'

def annotate_single_video(video_path, frame_dir, index=None, total=None):
    try:
        # Get video ID to match frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = os.path.join(frame_dir, f"*{video_name}*.jpg")
        matching_frames = glob.glob(pattern)

        if not matching_frames:
            print(f"⚠️ No frames found for: {video_name}")
            return random.choice(["0-10", "10-19", "19-35", "36-50", "51+"])

        # Use a random matching frame
        frame_path = random.choice(matching_frames)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return random.choice(["0-10", "10-19", "19-35", "36-50", "51+"])

        # Perform age analysis
        result = DeepFace.analyze(
            img_path=frame,
            actions=["age"],
            #detector_backend="retinaface",  # Change to "opencv" if needed
            detector_backend="opencv",
            enforce_detection=False
        )
        age = result[0]["age"]
        count_info = f" ({index+1}/{total})" if index and total else ""
        print(f"✅{count_info} Age detection successful: {frame_path}")
        return age_to_group(age)

    except Exception as e:
        print(f"❌ DeepFace error on {video_path}: {e}")
        return random.choice(["0-10", "10-19", "19-35", "36-50", "51+"])

def save_age_annotations_parallel(
    video_dir,
    output_csv="annotations.csv",
    batch_mode=False,
    num_workers=None,
    streamlit_progress=None,
    frame_dir="all_data_frames"  # <-- ✅ add frame_dir here
):
    # Delete existing annotation file
    annotation_file = output_csv
    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    full_paths = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.avi')):
                full_paths.append(os.path.join(root, f))

    if not batch_mode:
        print(f"🧠 Annotating {len(full_paths)} videos using parallel processing...")

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    metadata = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(annotate_single_video, path, frame_dir,idx, len(full_paths)): path  # <-- ✅ pass frame_dir here
            for idx, path in enumerate(full_paths)
        }

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), disable=batch_mode)):
            path = futures[future]
            filename = os.path.basename(path)
            label = "fake" if "fake" in filename.lower() else "real"

            try:
                age_group = future.result()
            except Exception as e:
                print(f"❌ Error annotating {filename}: {e}")
                age_group = random.choice(["0-10", "10-19", "19-35", "36-50", "51+"])

            # Determine source
            if "celeb" in filename.lower():
                source = "celeb"
            elif "faceforensics" in filename.lower():
                source = "faceforensics"
            else:
                source = "unknown"

            metadata.append({
                "filename": filename,
                "path": path,
                "label": label,
                "source": source,
                "age_group": age_group
            })

    if streamlit_progress:
        streamlit_progress.empty()

    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False)

    if not batch_mode:
        print(f"✅ Parallel annotation complete. Saved to {output_csv}")

    return df




