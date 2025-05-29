import os
import pandas as pd
import shutil
import random
import cv2
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


#------------------ Section for: Import REAL Images --------------

import os
import pandas as pd

def import_real_images(image_files, image_save_dir="all_data_videos/real_images", annotations_file="all_data_videos/annotations.csv"):
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    new_entries = []

    for img in image_files:
        filename = img.name
        image_path = os.path.join(image_save_dir, filename)
        
        with open(image_path, "wb") as f:
            f.write(img.read())

        try:
            age, gender, race, _ = filename.split("_", 3)
            age = int(age)
            if age <= 10:
                age_group = "0-10"
            elif age <= 19:
                age_group = "10-19"
            elif age <= 35:
                age_group = "19-35"
            elif age <= 50:
                age_group = "36-50"
            else:
                age_group = "51+"
        except:
            age, age_group = None, "unknown"

        new_entries.append({
            "filename": filename,
            "path": image_path,
            "label": "real",
            "source": "UTKFace",
            "age": age,
            "age_group": age_group
        })

    if os.path.exists(annotations_file):
        existing_df = pd.read_csv(annotations_file)
        # Drop rows where source is 'celeb' or 'faceforensics'
        existing_df = existing_df[~existing_df["source"].isin(["UTKFace"])]
    else:
        existing_df = pd.DataFrame()

    new_df = pd.DataFrame(new_entries)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["filename", "path"], inplace=True)
    combined_df.to_csv(annotations_file, index=False)

    return new_df, annotations_file


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




