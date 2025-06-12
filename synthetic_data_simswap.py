import subprocess
import os
import pandas as pd
import cv2
import uuid
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from insightface.app import FaceAnalysis

# Load InsightFace model once globally
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def analyze_face(image_path):
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if not faces:
        return None
    face = faces[0]

    gender = "male" if face.get('gender', 1) == 1 else "female"
    expression = "neutral" if face.get('expression', 0) == 0 else "non-neutral"
    yaw = face.get('pose', [0, 0])[0]
    pitch = face.get('pose', [0, 0])[1]
    brightness = cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]

    return {
        'gender': gender,
        'expression': expression,
        'yaw': yaw,
        'pitch': pitch,
        'brightness': brightness
    }

def match_best_video_by_face_attributes(image_path, frame_df, frame_dir="all_data_frames"):
    img_attrs = analyze_face(image_path)
    if not img_attrs:
        # print(f"[SKIP] Could not analyze source image: {image_path}")
        return None

    # print(f"\n🔍 Matching for: {os.path.basename(image_path)} (Gender: {img_attrs['gender']})")

    best_match = None
    best_score = float('inf')
    best_video_filename = None

    for _, row in frame_df.iterrows():
        if row['source'] not in ['celeb', 'faceforensics'] or row['label'] != 'real':
            continue

        video_filename = row['filename']
        video_base = os.path.splitext(video_filename)[0]

        # print(f"\n📽 Checking video: {video_filename}")
        matching_frames = [
            f for f in os.listdir(frame_dir)
            if video_base in f and f.endswith(('.jpg', '.png'))
        ]

        # print(f"🔎 Found {len(matching_frames)} frames for video: {video_filename}")
        if not matching_frames:
            continue

        for frame_file in matching_frames:
            frame_path = os.path.join(frame_dir, frame_file)
            if not os.path.exists(frame_path):
                # print(f"[WARN] Frame not found: {frame_path}")
                continue

            frame_attrs = analyze_face(frame_path)
            if not frame_attrs:
                # print(f"[WARN] No face detected in frame: {frame_file}")
                continue

            # print(f"🧠 Frame: {frame_file} | Frame Gender: {frame_attrs['gender']} | Expression: {frame_attrs['expression']}")

            if frame_attrs['gender'] != img_attrs['gender']:
                # print(f"[SKIP] Gender mismatch: Frame={frame_attrs['gender']} vs Image={img_attrs['gender']}")
                continue

            expr_penalty = 10 if frame_attrs['expression'] != 'neutral' else 0
            yaw_diff = abs(frame_attrs['yaw'] - img_attrs['yaw'])
            pitch_diff = abs(frame_attrs['pitch'] - img_attrs['pitch'])
            brightness_diff = abs(frame_attrs['brightness'] - img_attrs['brightness'])
            total_score = expr_penalty + yaw_diff + pitch_diff + 0.5 * brightness_diff

            # print(f"📏 Score = {total_score:.2f} (ExprPenalty={expr_penalty}, Yaw={yaw_diff:.1f}, Pitch={pitch_diff:.1f}, Bright={brightness_diff:.1f})")

            if total_score < best_score:
                best_score = total_score
                best_match = frame_path
                best_video_filename = video_filename
        
    video_path = os.path.join("all_data_videos", "real", best_video_filename)
    if os.path.exists(video_path):
        print(f"\n✅ Best match frame: {os.path.basename(best_match)} from {best_video_filename} | Score: {best_score:.2f}")
        return video_path
    else:
        print(f"\n[INFO] No good match found for: {os.path.basename(image_path)}")
        return None

    
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
        

def generate_synthetic_videos(image_df, streamlit_progress=None, st_module=None):
    import os
    import pandas as pd

    metadata_path = "all_data_videos/annotations.csv"
    balance_metadata_path = "final_output/balanced_metadata.csv"
    synthetic_dir = "all_data_videos/synthetic"
    os.makedirs(synthetic_dir, exist_ok=True)

    try:
        video_df = pd.read_csv(metadata_path)
        video_df = video_df[(video_df["label"] == "real") & (video_df["source"]!= "UTKFace")]
    except Exception as e:
        print(f"[ERROR] Could not load real videos for matching: {e}")
        return

    total = len(image_df)
    new_annotations = []

    for i, row in enumerate(image_df.itertuples(index=False), 1):
        try:
            image_path = os.path.join("all_data_videos/real_images", row.filename)
            matched_video = match_best_video_by_face_attributes(
                image_path=image_path,
                frame_df=video_df,
                frame_dir="all_data_frames"
            )

            if not matched_video or not os.path.exists(matched_video):
                print(f"[SKIP] No valid video match for image {row.filename}")
                continue

            #output_filename = f"{row.filename.split('.')[0]}_on_{os.path.basename(matched_video)}"
            image_base = row.filename.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            if image_base.startswith("utkface_data_real_"):
                image_base = image_base.replace("utkface_data_real_", "")

            matched_base = os.path.basename(matched_video)
            
            # Simplify matched video filename
            if "celeb_data_real_" in matched_base:
                matched_id = matched_base.replace("celeb_data_real_", "celeb_")
            elif "faceforensics_data_real_" in matched_base:
                matched_id = matched_base.replace("faceforensics_data_real_", "ff_")
            else:
                matched_id = os.path.splitext(matched_base)[0]
                
            output_filename = f"utkface_data_fake_{image_base}_on_{matched_id}"
            output_path = os.path.join(synthetic_dir, output_filename)

            simswap_single(image_path, matched_video, output_path)

            if not os.path.exists(output_path):
                print(f"[ERROR] Failed to generate: {output_filename}")
                continue

            new_annotations.append({
                "filename": output_filename,
                "path": output_path,
                "label": "fake",
                "source": "synthetic",
                "age": row.age,
                "age_group": row.age_group
            })

            if streamlit_progress and st_module:
                streamlit_progress.progress(i / total, text=f"Generating ({i}/{total})...")

        except Exception as e:
            print(f"[ERROR] Unexpected failure for row {i}: {e}")
            continue

    # Append new synthetic entries to final metadata
    if new_annotations:
        try:
            existing_df = pd.read_csv(balance_metadata_path)
            updated_df = pd.concat([existing_df, pd.DataFrame(new_annotations)], ignore_index=True)
            updated_df.to_csv(balance_metadata_path, index=False) 
            updated_df.to_csv(metadata_path, index=False)   
            print(f"[INFO] Metadata updated with {len(new_annotations)} synthetic videos.")
        except Exception as e:
            print(f"[ERROR] Could not update metadata: {e}")

#------------------------------------Synthetic Data creation---------------------------