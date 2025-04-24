import os
import shutil

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


def clean_import_directory(video_dir="all_data_videos", frames_dir="all_data_frames", annotation_file="annotations.csv"):
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

def clean_low_quality_videos(directory):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and not check_video_quality(path):
            os.remove(path)
            print(f"Removed low-quality video: {filename}")