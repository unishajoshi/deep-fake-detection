import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import json
from frame_extractor import extract_frames_from_combined_parallel
from synthetic_data_simswap import analyze_face

FAKE_BALANCE_TARGET = None

def preprocess_frames(input_dir, output_dir, target_size=(224, 224)):
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
            cv2.imwrite(os.path.join(output_dir, file), img)

    return output_dir

def balance_dataset(metadata_path):
    import json
    global FAKE_BALANCE_TARGET

    df = pd.read_csv(metadata_path)
    required_cols = {"age_group", "label", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    balanced_dfs = []
    real_group_sizes = {}

    # Step 1: Balance Celeb + FF++ for real and fake
    for label in ["real", "fake"]:
        subset = df[(df["label"] == label) & (df["source"] != "UTKFace")]
        target = int(subset.groupby("age_group").size().mean())

        if label == "fake":
            FAKE_BALANCE_TARGET = target

        for age_group, group in subset.groupby("age_group"):
            sampled = group.sample(n=target, replace=False, random_state=42) if len(group) >= target else group
            balanced_dfs.append(sampled)
            if label == "real":
                real_group_sizes[age_group] = len(sampled)

    # Step 2: Copy all UTKFace real images for synthetic use
    utk_real_df = df[(df["label"] == "real") & (df["source"] == "UTKFace")]
    utk_real_df.to_csv("final_output/synthetic_allocation.csv", index=False)
    print(f"[INFO] Saved all {len(utk_real_df)} UTKFace real images to synthetic_allocation.csv")

    # Step 3: Top-up real groups from UTKFace (not for synthetic)
    max_real_size = max(real_group_sizes.values(), default=0)
    for age_group, group in utk_real_df.groupby("age_group"):
        current_size = real_group_sizes.get(age_group, 0)
        needed = max_real_size - current_size
        if needed <= 0:
            continue

        scored_rows = []
        for row in group.itertuples(index=False):
            image_path = os.path.join("all_data_videos", "real_images", row.filename)
            try:
                attrs = analyze_face(image_path)
                if not attrs:
                    continue
                score = (
                    (1 if attrs["expression"] == "neutral" else 0) +
                    (1 - abs(attrs["yaw"]) / 90) +
                    (1 - abs(attrs["pitch"]) / 90) +
                    (1 - abs(attrs["brightness"] - 128) / 128)
                )
                scored_rows.append((row, score))
            except Exception as e:
                print(f"[WARN] Scoring failed for {row.filename}: {e}")
                continue

        top_rows = [x[0] for x in sorted(scored_rows, key=lambda x: x[1], reverse=True)[:needed]]
        if top_rows:
            balanced_dfs.append(pd.DataFrame(top_rows))

    # Step 4: Final save
    df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    df_balanced.to_csv("final_output/balanced_metadata.csv", index=False)

    with open("final_output/balance_config.json", "w") as f:
        json.dump({"fake_balance_target": FAKE_BALANCE_TARGET}, f)

    return df_balanced

def export_balanced_dataset(balanced_csv="final_output/balanced_annotations.csv", output_dir="final_output"):
    video_output_dir = os.path.join(output_dir, "age_diverse_videos")
    df = pd.read_csv(balanced_csv)
    for label in ["real", "fake"]:
        os.makedirs(os.path.join(video_output_dir, label), exist_ok=True)

    copied, missing = 0, []
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
    
    if mode == "balanced":
        utkface_real = df[(df["source"] == "UTKFace") & (df["label"] == "real")]
        if not utkface_real.empty:
            utkface_real = utkface_real.copy()
            utkface_real["frame"] = utkface_real["filename"] 
            utkface_real["path"] = utkface_real["path"]
            utkface_real_subset = utkface_real[["frame", "path", "label", "age_group", "source"]]
            frame_df = pd.concat([frame_df, utkface_real_subset], ignore_index=True)

    frame_df.to_csv(output_csv, index=False)
    print(f"✅ Saved frame-level annotations: {output_csv} ({len(frame_df)} rows)")

    #---- update frame_level_annotations_source with rows from frame_level_annotations----------
    
    if mode == "balanced":
        source_df = pd.read_csv("final_output/frame_level_annotations_source.csv")
        balanced_df = pd.read_csv("final_output/frame_level_annotations.csv")
        filtered_source = source_df[
            (source_df["source"].isin(["celeb", "faceforensics"])) &
            (source_df["frame"].isin(balanced_df["frame"]))
        ]
        filtered_source.to_csv("final_output/frame_level_annotations_source.csv", index=False)
        print(f"✅ Saved frame-level annotations for celeb and faceforensics: {source_df} ({len(source_df)} rows)")

def merge_synthetic_into_balanced_annotations(
    balanced_annotations_path="final_output/balanced_annotations.csv",
    balanced_metadata_path="final_output/balanced_metadata.csv"
):
    try:
        meta_df = pd.read_csv(balanced_metadata_path)
        anno_df = pd.read_csv(balanced_annotations_path) if os.path.exists(balanced_annotations_path) else pd.DataFrame()
        synthetic_df = meta_df[meta_df["source"] == "synthetic"]
        new_entries = synthetic_df[~synthetic_df["filename"].isin(anno_df.get("filename", []))]
        if not new_entries.empty:
            combined = pd.concat([anno_df, new_entries], ignore_index=True)
            combined.to_csv(balanced_annotations_path, index=False)
            print(f"✅ Merged {len(new_entries)} synthetic entries")
    except Exception as e:
        print(f"❌ Error merging synthetic data: {e}")

def generate_synthetic_frame_annotations(
    synthetic_dir="all_data_videos/synthetic",
    frame_dir="all_data_frames",
    output_csv="final_output/frame_level_annotations.csv",
    metadata_csv="final_output/balanced_metadata.csv",
    frame_rate=10
):
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
    video_map = {os.path.splitext(row["filename"])[0]: row for _, row in synthetic_df.iterrows()}
    frame_annotations = []

    for frame_path in glob(os.path.join(frame_dir, "fake_*.jpg")):
        frame_name = os.path.basename(frame_path)
        matched = next((video_map[vn] for vn in video_map if vn in frame_name), None)
        if matched is not None:
            frame_annotations.append({
                "frame": frame_name,
                "path": frame_path,
                "label": "fake",
                "age_group": matched["age_group"],
                "source": "synthetic"
            })

    if frame_annotations:
        frame_df = pd.DataFrame(frame_annotations)
        existing = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()
        updated = pd.concat([existing, frame_df], ignore_index=True)
        updated.to_csv(output_csv, index=False)
        print(f"✅ Appended {len(frame_df)} synthetic frames")