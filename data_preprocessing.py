import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

def preprocess_frames(input_dir, output_dir, target_size=(224, 224)):
    """
    Resize and clean frame images for model input.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(os.listdir(input_dir)):
        if file.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(input_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, img)

    return output_dir


def balance_dataset(metadata_path, method="oversample"):
    """
    Balance dataset by both age group and label (real/fake).
    Ensures each age group has an equal number of real and fake samples.
    """
    df = pd.read_csv(metadata_path)
    balanced_dfs = []

    for (age_group), group_df in df.groupby("age_group"):
        real_df = group_df[group_df["label"] == 0]
        fake_df = group_df[group_df["label"] == 1]

        if method == "oversample":
            max_len = max(len(real_df), len(fake_df))
            real_balanced = real_df.sample(max_len, replace=True)
            fake_balanced = fake_df.sample(max_len, replace=True)
        elif method == "undersample":
            min_len = min(len(real_df), len(fake_df))
            real_balanced = real_df.sample(min_len, replace=False)
            fake_balanced = fake_df.sample(min_len, replace=False)
        else:
            raise ValueError("Method must be 'oversample' or 'undersample'.")

        balanced_dfs.append(pd.concat([real_balanced, fake_balanced]))

    df_balanced = pd.concat(balanced_dfs).sample(frac=1).reset_index(drop=True)
    df_balanced.to_csv("balanced_metadata.csv", index=False)
    return df_balanced


def export_balanced_dataset(balanced_csv="balanced_annotations.csv", output_dir="final_output"):
    """
    Copy videos listed in balanced_annotations.csv to final output directory.
    Structure: final_output/age_diverse_videos/{real,fake}
    Also saves balanced_annotations.csv in final_output/
    """
    video_output_dir = os.path.join(output_dir, "age_diverse_videos")
    df = pd.read_csv(balanced_csv)

    # Create output directories
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(video_output_dir, label), exist_ok=True)

    copied = 0
    missing = []

    for _, row in df.iterrows():
        src_path = row["path"]
        label = row["label"]
        filename = os.path.basename(src_path)
        dest_path = os.path.join(video_output_dir, label, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            missing.append(filename)

    # Save a copy of the balanced CSV
    df.to_csv(os.path.join(output_dir, "balanced_annotations.csv"), index=False)

    return copied, missing, output_dir


def generate_frame_level_annotations(
    frame_dir="all_data_frames",
    mode="annotated"
):
    # Auto-select input/output files based on mode
    if mode == "annotated":
        video_csv = "annotations.csv"
        output_csv = "frame_level_annotations_original.csv"
    elif mode == "balanced":
        video_csv = "balanced_annotations.csv"
        output_csv = "frame_level_annotations.csv"
    else:
        raise ValueError("mode must be either 'annotated' or 'balanced'")
    df = pd.read_csv(video_csv)
    metadata = []

    # Create a mapping from video name (without extension) to its row
    video_map = {
        os.path.splitext(row["filename"])[0]: row
        for _, row in df.iterrows()
    }

    # Loop through all frames and match them with video metadata
    for frame_path in tqdm(glob(os.path.join(frame_dir, "*.jpg")), desc="Generating frame annotations"):
        frame_name = os.path.basename(frame_path)
        matched_video = None

        for video_name in video_map:
            if video_name in frame_name:
                matched_video = video_map[video_name]
                break

        if matched_video is not None:
            metadata.append({
                "frame": frame_name,
                "path": frame_path,
                "label": matched_video["label"],
                "age_group": matched_video["age_group"],
                "source": matched_video["source"]
            })

    frame_df = pd.DataFrame(metadata)
    frame_df.to_csv(output_csv, index=False)
    print(f"✅ Saved frame-level annotations: {output_csv} ({len(frame_df)} rows)")