import os
import pandas as pd
import shutil
import random
import cv2
from tqdm import tqdm
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

#----------import real image from zip-----------------------

def import_real_images_zip(image_files, image_save_dir="all_data_videos/real_images", annotations_file="all_data_videos/annotations.csv"):
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    new_entries = []

    for img in image_files:
        try:
            original_name = img.name
            filename = f"utkface_real_{original_name}"
            image_path = os.path.join(image_save_dir, filename)

            with open(image_path, "wb") as f:
                f.write(img.read())

            # Verify image integrity
            if cv2.imread(image_path) is None:
                continue

            parsed_name = original_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            age, gender, race, _ = parsed_name.split("_", 3)
            age = int(age)

            if age < 10:
                age_group =  '0-10'
            elif age < 19:
                age_group =  '10-18'
            elif age < 36:
                age_group =  '19-35'
            elif age < 51:
                age_group =  '36-50'
            else:
                age_group =  '51+'

        

        except Exception:
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
        existing_df = existing_df[~existing_df["source"].isin(["UTKFace"])]
    else:
        existing_df = pd.DataFrame()

    new_df = pd.DataFrame(new_entries)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["filename", "path"], inplace=True)
    combined_df.to_csv(annotations_file, index=False)

    return new_df, annotations_file


def import_real_images_from_zip(uploaded_zip, extract_dir="all_data_videos/real_images"):
    os.makedirs(extract_dir, exist_ok=True)

    # Extract ZIP to temporary folder
    with zipfile.ZipFile(uploaded_zip) as zip_ref:
        zip_ref.extractall(extract_dir)

    # Collect valid image-like objects
    image_files = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                with open(full_path, "rb") as img_file:
                    content = img_file.read()

                # Skip if empty
                if not content:
                    continue

                image_like = type("UploadedFile", (), {
                    "name": f,
                    "read": lambda c=content: c
                })()
                image_files.append(image_like)

    if not image_files:
        return None, None

    return import_real_images_zip(image_files, image_save_dir=extract_dir)
    
#------------------ Section for: Import REAL Images --------------

def import_real_images(image_files, image_save_dir="all_data_videos/real_images", annotations_file="all_data_videos/annotations.csv"):
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    new_entries = []

    for img in image_files:
        try:
            original_name = img.name
            filename = f"utkface_real_{original_name}"
            image_path = os.path.join(image_save_dir, filename)

            # Safe write with file check
            img_bytes = img.read()
            with open(image_path, "wb") as f:
                f.write(img_bytes)

            # Optional: verify if it's a valid image
            image_check = cv2.imread(image_path)
            if image_check is None:
                st.warning(f"⚠️ Failed to read image: {original_name}. Skipping.")
                continue

            # Parse filename for metadata
            parsed_name = original_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            age, gender, race, _ = parsed_name.split("_", 3)
            age = int(age)

            if age < 10:
                age_group =  '0-10'
            elif age < 19:
                age_group =  '10-18'
            elif age < 36:
                age_group =  '19-35'
            elif age < 51:
                age_group =  '36-50'
            else:
                age_group =  '51+'

        except Exception as e:
            st.warning(f"⚠️ Error processing image `{img.name}`: {e}")
            continue

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
        existing_df = existing_df[~existing_df["source"].isin(["UTKFace"])]
    else:
        existing_df = pd.DataFrame()

    new_df = pd.DataFrame(new_entries)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["filename", "path"], inplace=True)
    combined_df.to_csv(annotations_file, index=False)

    return new_df, annotations_file

def load_images_from_dir(image_dir):
    image_files = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, fname)
                with open(path, "rb") as f:
                    file_like = type("UploadedFile", (), {
                        "name": fname,
                        "read": lambda f=f: f.read()
                    })()
                    image_files.append(file_like)
    return image_files

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




