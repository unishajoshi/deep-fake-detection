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
import gc
from itertools import islice


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
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = os.path.join(frame_dir, f"*{video_name}*.jpg")
        matching_frames = glob.glob(pattern)

        if not matching_frames:
            print(f"⚠️ No frames found for: {video_name}")
            return None, None

        # Randomly select a frame
        frame_path = random.choice(matching_frames)
        frame = cv2.imread(frame_path)

        if frame is None or frame.size == 0:
            print(f"❌ Frame is empty or unreadable: {frame_path}")
            return None, None

        # Validate image size
        h, w = frame.shape[:2]
        if h < 60 or w < 60:
            print(f"⚠️ Frame too small for analysis: {frame_path}")
            return None, None

        # Run DeepFace analysis with enforced safety
        result = DeepFace.analyze(
            img_path=frame,
            actions=["age"],
            detector_backend="opencv",
            enforce_detection=False
        )

        if not result or "age" not in result[0]:
            print(f"⚠️ Age not detected for: {frame_path}")
            return None, None

        real_age = result[0]["age"]
        age_group = age_to_group(real_age)

        count_info = f" ({index+1}/{total})" if index is not None and total else ""
        print(f"✅{count_info} Age detection successful: {frame_path} → Age: {real_age}")
        
        gc.collect()

        return real_age, age_group

    except Exception as e:
        print(f"❌ DeepFace error on {video_path}: {e}")
        gc.collect()
        return None, None



def chunked_iterable(iterable, size):
    """Yield successive chunks from iterable of specified size."""
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def save_age_annotations_parallel(
    video_dir,
    output_csv="all_data_videos/annotations.csv",
    batch_mode=False,
    num_workers=None,
    streamlit_progress=None,
    frame_dir="all_data_frames"
):
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
    else:
        existing_df = pd.DataFrame()

    full_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(video_dir)
        for f in files if f.endswith(('.mp4', '.avi'))
    ]

    if not batch_mode:
        print(f"🧠 Annotating {len(full_paths)} videos using parallel processing...")

    if num_workers is None:
        num_workers = min(4, max(1, os.cpu_count() - 1))  # Conservative thread pool

    metadata = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch in chunked_iterable(enumerate(full_paths), 10):  # Process in batches of 10
            futures = {
                executor.submit(annotate_single_video, path, frame_dir, idx, len(full_paths)): path
                for idx, path in batch
            }

            for future in as_completed(futures):
                path = futures[future]
                filename = os.path.basename(path)
                label = "fake" if "fake" in filename.lower() else "real"
                source = "celeb" if "celeb" in filename.lower() else (
                         "faceforensics" if "faceforensics" in filename.lower() else "unknown")

                try:
                    real_age, age_group = future.result()
                    if real_age is None and age_group is None:
                        print(f"⚠️ Skipping annotation for: {filename} (no valid frame)")
                        continue
                except Exception as e:
                    print(f"❌ Error annotating {filename}: {e}")
                    continue

                metadata.append({
                    "filename": filename,
                    "path": path,
                    "label": label,
                    "source": source,
                    "age": real_age,
                    "age_group": age_group
                })

    if streamlit_progress:
        streamlit_progress.empty()

    # Save and clean memory
    new_df = pd.DataFrame(metadata)
    df = pd.concat([existing_df, new_df], ignore_index=True)
    df.drop_duplicates(subset=["filename", "path"], inplace=True)
    df.to_csv(output_csv, index=False)

    del metadata, new_df, df
    gc.collect()

    if not batch_mode:
        print(f"✅ Annotation complete. Saved to {output_csv}")

    return pd.read_csv(output_csv)


