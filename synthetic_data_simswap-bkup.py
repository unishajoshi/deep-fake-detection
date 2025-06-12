import subprocess
import os
import pandas as pd
import cv2
import uuid
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


def align_face(image_path, save_path):
    img = Image.open(image_path).convert('RGB')
    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device)
    face = mtcnn(img_bgr)

    if face is None:
        raise ValueError(f"No face detected in image: {image_path}")

    if isinstance(face, torch.Tensor):
        face = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    elif isinstance(face, np.ndarray):
        face = face.astype(np.uint8)
    else:
        raise TypeError(f"[ERROR] Unexpected face output type: {type(face)}")

    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    success = cv2.imwrite(save_path, face_bgr)

    if not success:
        raise IOError(f"[ERROR] Failed to write aligned image: {save_path}")

    print(f"[DEBUG] Aligned face saved: {save_path}")
    
def simswap_single(source_path, target_path, output_path):
    original_dir = os.getcwd()
    simswap_dir = "SimSwap"
    python_executable = "python"  # or "python3" on Linux/macOS

    # Create aligned temp image
    #aligned_dir = "temp_aligned"
    #os.makedirs(aligned_dir, exist_ok=True)
    #aligned_img_path = os.path.join(aligned_dir, f"{uuid.uuid4().hex}.jpg")
    #print(f"[DEBUG] Aligned face image: {aligned_img_path}")

    try:
        # Step 1: Face alignment
        #align_face(source_path, aligned_img_path)
        #print(f"[DEBUG] Aligned face complete image: {aligned_img_path}")
        # Resolve absolute paths
        abs_target = os.path.abspath(target_path)
        abs_output = os.path.abspath(output_path)
        abs_simswap = os.path.abspath(os.path.join(simswap_dir, "run_simswap.py"))
        #abs_aligned = os.path.abspath(aligned_img_path)
        abs_source= os.path.abspath(source_path)
        

        command = [
            python_executable,
            abs_simswap,
            "--pic_a_path", abs_source,
            "--video_path", abs_target,
            "--output_path", abs_output,
            "--use_mask"
        ]

        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        subprocess.run(command, check=True)
        print(f"[INFO] SimSwap completed: {os.path.basename(abs_output)}")

    except Exception as e:
        print(f"[ERROR] SimSwap failed: {e}")

    #finally:
     #   if os.path.exists(aligned_img_path):
      #      os.remove(aligned_img_path)
       # os.chdir(original_dir)
        
def find_closest_video(image_age, video_df):
    video_df = video_df.copy()
    video_df = video_df[pd.notnull(video_df["age"])]
    video_df["age"] = pd.to_numeric(video_df["age"], errors="coerce")
    video_df = video_df.dropna(subset=["age"])

    if video_df.empty:
        return None

    video_df["age_diff"] = (video_df["age"] - image_age).abs()
    closest = video_df.loc[video_df["age_diff"].idxmin()]

    # Construct path if needed
    if "path" in closest:
        path = closest["path"]
    else:
        path = os.path.join("all_data_videos", "celeb", closest["filename"])

    return path if os.path.exists(path) else None

def generate_synthetic_videos(image_df, streamlit_progress=None, st_module=None):
    annotation_file = "all_data_videos/annotations.csv"
    os.makedirs("all_data_videos/synthetic", exist_ok=True)

    try:
        video_df = pd.read_csv(annotation_file)
        video_df = video_df[(video_df["label"] == "real") & (video_df["source"] == "celeb")]
        print(video_df[['filename', 'age', 'age_group']].head())
    except Exception as e:
        print(f"[ERROR] Could not load real videos for matching: {e}")
        return

    total = len(image_df)
    print(f"[INFO] Starting synthetic generation for {total} images")
    generated_records = []
    new_annotations = []

    for i, row in enumerate(image_df.itertuples(index=False), 1):
        try:
            image_path = os.path.join("all_data_videos/real_images", row.filename)
            matched_video = find_closest_video(row.age, video_df)

            if not matched_video or not os.path.exists(matched_video):
                print(f"[SKIP] No valid video match for image {row.filename}")
                continue

            output_filename = f"{row.filename.split('.')[0]}_on_{os.path.basename(matched_video)}"
            output_path = os.path.join("all_data_videos/synthetic", output_filename)

            simswap_single(image_path, matched_video, output_path)

            if not os.path.exists(output_path):
                print(f"[ERROR] Failed to generate synthetic video at: {output_path}")
                continue

            generated_records.append({
                "Age Group": row.age_group,
                "Source Image": row.filename,
                "Matched Video": os.path.basename(matched_video),
                "Generated Video": output_filename
            })

            new_annotations.append({
                "filename": output_filename,
                "path": output_path,
                "label": "fake",
                "source": "synthetic",
                "age": row.age,
                "age_group": row.age_group
            })

            if st_module and streamlit_progress:
                streamlit_progress.progress(i / total, text=f"Generating ({i}/{total})...")

        except Exception as e:
            print(f"[ERROR] Unexpected failure for row {i}: {e}")
            continue

#------------------------------------Synthetic Data creation---------------------------

def batch_simswap(source_images_dir, target_videos_dir, output_dir, metadata_path):
    import os
    import pandas as pd
    
    metadata_records = []
    
    for img_file in os.listdir(source_images_dir):
        if not img_file.endswith(".jpg"):
            continue
        source_path = os.path.join(source_images_dir, img_file)
        
        for video_file in os.listdir(target_videos_dir):
            if not video_file.endswith(".mp4"):
                continue
            target_path = os.path.join(target_videos_dir, video_file)
            output_path = os.path.join(output_dir, f"{img_file.split('.')[0]}_on_{video_file}")
            
            # Call SimSwap (this will vary depending on your setup)
            simswap_single(source_path, target_path, output_path)
            
            metadata_records.append({
                'source_image': img_file,
                'target_video': video_file,
                'generated_video': os.path.basename(output_path),
                'label': 'fake',
                'source': 'simswap'
            })
    
    pd.DataFrame(metadata_records).to_csv(metadata_path, index=False)
