import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from frame_extractor import extract_frames_from_combined_parallel
import json
from synthetic_data_simswap import analyze_face

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


def balance_dataset_bkup(metadata_path):
    """
    Balance dataset by age group and label (real/fake), excluding 'UTKFace'.
    Ensures each age group has equal number of real and fake samples.
    """
    df = pd.read_csv(metadata_path)

    if "age_group" not in df.columns or "label" not in df.columns or "source" not in df.columns:
        raise ValueError("Dataset must contain 'age_group', 'label', and 'source' columns.")

    # Separate UTKFace and non-UTKFace data
    utkface_df = df[df["source"] == "UTKFace"]
    other_df = df[df["source"] != "UTKFace"]

    balanced_dfs = []

    for age_group, group_df in other_df.groupby("age_group"):
        real_df = group_df[group_df["label"] == "real"]
        fake_df = group_df[group_df["label"] == "fake"]

        min_len = min(len(real_df), len(fake_df))
        if min_len == 0:
            continue

        real_balanced = real_df.sample(min_len, replace=False, random_state=42)
        fake_balanced = fake_df.sample(min_len, replace=False, random_state=42)

        balanced_dfs.append(pd.concat([real_balanced, fake_balanced]))

    if not balanced_dfs:
        raise ValueError("No balanced groups could be created.")

    # Combine balanced data and untouched UTKFace data
    df_balanced = pd.concat(balanced_dfs + [utkface_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced.to_csv("final_output/balanced_metadata.csv", index=False)
    return df_balanced

# Global variable to store fake balance target for reuse in synthetic data generation
FAKE_BALANCE_TARGET = None

def balance_dataset(metadata_path):
    """
    Balance dataset by age group and label (real/fake).
    - For fake: undersample Celeb/FF++ to average, and store target globally.
    - For real: balance Celeb/FF++ to average, then use UTKFace to equalize real age group sizes.
    """
    global FAKE_BALANCE_TARGET

    df = pd.read_csv(metadata_path)

    required_cols = {"age_group", "label", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    balanced_dfs = []
    real_group_sizes = {}

    # --- Step 1: Balance Celeb + FF++ for both labels ---
    for label in ["real", "fake"]:
        subset = df[(df["label"] == label) & (df["source"] != "UTKFace")]
        group_counts = subset.groupby("age_group").size()
        if group_counts.empty:
            continue

        target = int(group_counts.mean())

        if label == "fake":
            FAKE_BALANCE_TARGET = target  # Store for use in synthetic generation

        for age_group, group in subset.groupby("age_group"):
            sampled = group.sample(n=target, replace=False, random_state=42) if len(group) >= target else group.copy()
            balanced_dfs.append(sampled)

            if label == "real":
                real_group_sizes[age_group] = len(sampled)

    # --- Step 2: Use UTKFace to top-up real groups to match max real group size ---
    max_real_size = max(real_group_sizes.values(), default=0)
    utk_real_df = df[(df["label"] == "real") & (df["source"] == "UTKFace")]
    
    for age_group, group in utk_real_df.groupby("age_group"):
        current_size = real_group_sizes.get(age_group, 0)
        needed = max_real_size - current_size

        if needed <= 0:
            continue

        # Score each sample using analyze_face
        scored_rows = []
        for row in group.itertuples(index=False):
            image_path = os.path.join("all_data_videos", "real_images", row.filename)
            try:
                attrs = analyze_face(image_path)
                if not attrs:
                    continue
                # Scoring: prefer neutral, low yaw/pitch, and mid-range brightness
                score = (
                    (1 if attrs["expression"] == "neutral" else 0) +
                    (1 - abs(attrs["yaw"]) / 90) +
                    (1 - abs(attrs["pitch"]) / 90) +
                    (1 - abs(attrs["brightness"] - 128) / 128)
                )
                scored_rows.append((row, score))
            except Exception as e:
                print(f"[WARN] Failed to score {row.filename}: {e}")
                continue

        # Sort and select top 'needed'
        scored_rows.sort(key=lambda x: x[1], reverse=True)
        top_rows = [x[0] for x in scored_rows[:needed]]

        if not top_rows:
            continue

        sampled = pd.DataFrame(top_rows)
        balanced_dfs.append(sampled)

        if len(top_rows) < needed:
            print(f"⚠️ UTKFace: age group '{age_group}' underfilled — needed {needed}, selected {len(top_rows)}")
            
            
    # --- Step 3: Finalize and save ---
    if not balanced_dfs:
        raise ValueError("No data was selected for balancing.")

    df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    df_balanced.to_csv("final_output/balanced_metadata.csv", index=False)
    
    with open("final_output/balance_config.json", "w") as f:
        print(f"[INFO] Saving FAKE_BALANCE_TARGET = {FAKE_BALANCE_TARGET}")
        json.dump({"fake_balance_target": FAKE_BALANCE_TARGET}, f)
    
    return df_balanced

def export_balanced_dataset(balanced_csv="final_output/balanced_annotations.csv", output_dir="final_output"):
    """
    Copy videos listed in final_output/balanced_annotations.csv to final output directory.
    Structure: final_output/age_diverse_videos/{real,fake}
    Also saves final_output/balanced_annotations.csv in final_output/
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
        video_csv = "all_data_videos/annotations.csv"
        output_csv = "final_output/frame_level_annotations_source.csv"
    elif mode == "balanced":
        video_csv = "final_output/balanced_annotations.csv"
        output_csv = "final_output/frame_level_annotations.csv"
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
    
 ## ---------------------FOr Synthetic Data-----------------------------
def merge_synthetic_into_balanced_annotations(
    balanced_annotations_path="final_output/balanced_annotations.csv",
    balanced_metadata_path="final_output/balanced_metadata.csv"
):
    try:
        meta_df = pd.read_csv(balanced_metadata_path)
        anno_df = pd.read_csv(balanced_annotations_path) if os.path.exists(balanced_annotations_path) else pd.DataFrame()

        synthetic_df = meta_df[meta_df["source"] == "synthetic"]
        if not synthetic_df.empty:
            new_entries = synthetic_df[~synthetic_df["filename"].isin(anno_df.get("filename", []))]
            combined = pd.concat([anno_df, new_entries], ignore_index=True)
            combined.to_csv(balanced_annotations_path, index=False)
            print(f"✅ Merged {len(new_entries)} synthetic entries into balanced_annotations.csv")
        else:
            print("ℹ️ No synthetic entries found to merge.")
    except Exception as e:
        print(f"❌ Error merging synthetic data into balanced_annotations: {e}")

def generate_synthetic_frame_annotations(
    synthetic_dir="all_data_videos/synthetic",
    frame_dir="all_data_frames",
    output_csv="final_output/frame_level_annotations.csv",
    metadata_csv="final_output/balanced_metadata.csv",
    frame_rate=10
):
    import glob
    from frame_extractor import extract_frames_from_combined_parallel
    extract_frames_from_combined_parallel(
        combined_dir=synthetic_dir,
        frame_output=frame_dir,
        frame_rate=frame_rate,
        overwrite=False,
        streamlit_mode=False,
        batch_mode=True
    )

    df = pd.read_csv(metadata_csv)
    synthetic_df = df[df["source"] == "synthetic"]

    video_map = {
        os.path.splitext(row["filename"])[0]: row
        for _, row in synthetic_df.iterrows()
    }

    frame_annotations = []
    for frame_path in glob.glob(os.path.join(frame_dir, "fake_*.jpg")):
        frame_name = os.path.basename(frame_path)
        for video_name in video_map:
            if video_name in frame_name:
                row = video_map[video_name]
                frame_annotations.append({
                    "frame": frame_name,
                    "path": frame_path,
                    "label": "fake",
                    "age_group": row["age_group"],
                    "source": "synthetic"
                })
                break

    if not frame_annotations:
        print("⚠️ No matching synthetic frames found.")
        return

    frame_df = pd.DataFrame(frame_annotations)
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        updated = pd.concat([existing, frame_df], ignore_index=True)
    else:
        updated = frame_df

    updated.to_csv(output_csv, index=False)
    print(f"✅ Appended {len(frame_df)} synthetic frames to {output_csv}")