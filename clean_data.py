import os
import shutil
import cv2
import pandas as pd

#---------------------section: Cleanup the files and folders---------------------------
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")


import os
import glob


def clean_videos_and_files(video_dir="all_data_videos", frames_dir="all_data_frames", annotation_file="annotations.csv"):
    # Define target file extensions
    extensions = ("*.csv", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.jpg", "*.jpeg", "*.png")

    # Recursively find and delete files with the given extensions
    for ext in extensions:
        for file_path in glob.glob(f"**/{ext}", recursive=True):
            # Skip files inside the SimSwap directory
            if "SimSwap" in file_path:
                continue
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"✅ Removed: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to remove {file_path}: {e}")
                    
    print("✅ All video folders, CSV files, and frame folders cleaned (SimSwap excluded).")
    return True
   
    
def clean_spreadsheet_files(file_path="final_output"):
    # Clear real and fake video folders
    for subfolder in ["real", "fake"]:
        sub_dir = os.path.join(video_dir, subfolder)
        if os.path.exists(sub_dir):
            clear_folder(sub_dir)
        else:
            os.makedirs(sub_dir, exist_ok=True)

    # Delete annotation file
    if os.path.exists(annotation_file):
        os.remove(annotation_file)
        print(f"✅ Removed: {annotation_file}")

    # Clear extracted frames
    clear_folder(frames_dir)

    print("✅ All imports and frame folders cleaned.")
    return True

#------------------ Section for: CLEANING LOW QUALITY VIDEO  --------------

def check_video_quality(video_path, min_width= 224, min_height=224):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width >= min_width and height >= min_height

def clean_low_quality_videos(directory, min_width=224, min_height=224):
    removed_files = []
    valid_extensions = (".mp4", ".avi", ".mov")

    if not os.path.exists(directory):
        return removed_files  # return empty list if path doesn't exist

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and filename.lower().endswith(valid_extensions):
            try:
                if not check_video_quality(path, min_width, min_height):
                    os.remove(path)
                    removed_files.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return removed_files
    
#--------------- Section for: Removing pdf files----------------------

def remove_pdf_files(folder="final_output"):
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    for file in pdf_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"[ERROR] Could not remove {file}: {e}")

#--------------- Section for: Removing loq quality image files----------------------

def is_valid_image(image_path, min_width=224, min_height=224):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        height, width = img.shape[:2]
        return width >= min_width and height >= min_height
    except:
        return False

def clean_low_quality_images(image_dir, annotations_file, min_width=224, min_height=224):
    if not os.path.exists(annotations_file):
        return []

    df = pd.read_csv(annotations_file)
    to_remove = []

    for index, row in df.iterrows():
        path = row.get("path")
        if row.get("source") == "UTKFace" and isinstance(path, str) and path.startswith(image_dir):
            if not is_valid_image(path, min_width, min_height):
                try:
                    os.remove(path)
                    to_remove.append(index)
                    print(f"Removed low-quality image: {path}")
                except Exception as e:
                    print(f"Error removing image {path}: {e}")

    df.drop(index=to_remove, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(annotations_file, index=False)

    return to_remove