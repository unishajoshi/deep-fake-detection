import subprocess
import os
import pandas as pd
import cv2
import uuid
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Load InsightFace model once globally
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0)
import subprocess
    
def analyze_face(image_path):
    import cv2
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    if not faces:
        return None

    face = faces[0]

    gender = "male" if face.get('gender', 1) == 1 else "female"

    # Full expression mapping
    expression_map = {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "surprise",
        4: "anger",
        5: "disgust",
        6: "fear"
    }
    expression_code = face.get('expression', 0)
    expression = expression_map.get(expression_code, "unknown")

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

@lru_cache(maxsize=None)
def cached_analyze_face(path):
    return analyze_face(path)

def match_best_video_by_face_attributes(image_path, frame_df, frame_index, frame_dir="all_data_frames"):


    img_attrs = cached_analyze_face(image_path)
    if not img_attrs:
        return None

    candidate_rows = frame_df[
        (frame_df['source'].isin(['celeb', 'faceforensics'])) &
        (frame_df['label'] == 'real')
    ]

    best_match = None
    best_score = float('-inf')  # HIGHER is better now
    best_video_filename = None

    for _, row in candidate_rows.iterrows():
        video_filename = row['filename']
        video_base = os.path.splitext(video_filename)[0]

        # Get random 5 frames for matching
        if video_base not in frame_index.groups:
            continue  # Skip if video has no precomputed frame data
        
        candidate_group = frame_index.get_group(video_base)
        candidate_frames = candidate_group.sample(n=min(5, len(candidate_group)), random_state=42)

        def compute_score(frame_row):
            if frame_row['gender'] != img_attrs['gender']:
                return None, float('-inf')

            # Score each attribute: higher = better match
            expression_score = 1 if frame_row['expression'] == 'neutral' else 0
            yaw_score = 1 - min(abs(frame_row['yaw'] - img_attrs['yaw']) / 90, 1)
            pitch_score = 1 - min(abs(frame_row['pitch'] - img_attrs['pitch']) / 90, 1)
            brightness_diff = abs(frame_row['brightness'] - img_attrs['brightness']) / 255
            brightness_score = 1 - min(brightness_diff, 1)

            total_score = (
                1.0 * expression_score +
                1.5 * yaw_score +
                1.5 * pitch_score +
                1.0 * brightness_score
            )
            return frame_row['frame_file'], total_score

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(compute_score, candidate_frames.to_dict(orient='records')))

        for frame_file, score in results:
            if score > best_score:
                best_score = score
                best_match = frame_file
                best_video_filename = video_filename

    if best_match and best_video_filename:
        video_path = os.path.join("all_data_videos", "real", best_video_filename)
        if os.path.exists(video_path):
            print(f"\n✅ Best match frame: {os.path.basename(best_match)} from {best_video_filename} | Score: {best_score:.2f}")
            return video_path, best_score

    print(f"\n[INFO] No good match found for: {os.path.basename(image_path)}")
    return None, None

def precompute_frame_attributes(frame_dir="all_data_frames", output_csv="final_output/precomputed_frame_attrs.csv", st_module=None):
    results = []
    for fname in tqdm(os.listdir(frame_dir)):
        if st_module and st_module.session_state.get("stop_generation", False):
            print("🛑 Precompute stopped by user.")
            break

        if not fname.endswith((".jpg", ".png")):
            continue
        frame_path = os.path.join(frame_dir, fname)
        attrs = analyze_face(frame_path)
        if attrs:
            attrs["frame_file"] = fname
            results.append(attrs)

    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"✅ Precomputed attributes saved to: {output_csv}")

    
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
    
def simswap_single(source_path, target_path, output_path, st_module=None):
    simswap_dir = "SimSwap"
    python_executable = "python"

    abs_target = os.path.abspath(target_path)
    abs_output = os.path.abspath(output_path)
    abs_simswap = os.path.abspath(os.path.join(simswap_dir, "run_simswap.py"))
    abs_source = os.path.abspath(source_path)

    command = [
        python_executable,
        abs_simswap,
        "--pic_a_path", abs_source,
        "--video_path", abs_target,
        "--output_path", abs_output,
        "--use_mask"
    ]

    try:
        proc = subprocess.Popen(command)
        if st_module:
            st_module.session_state["simswap_process"] = proc
        return proc
    except Exception as e:
        print(f"[ERROR] SimSwap failed: {e}")
        return None
        

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
    match_log = [] 
    
    precomputed_path = "final_output/precomputed_frame_attrs.csv"
    if not os.path.exists(precomputed_path) or os.path.getsize(precomputed_path) == 0:
        print("🔍 precomputed_frame_attrs.csv not found or empty. Generating now...")
        precompute_frame_attributes(st_module=st_module)
    else:
        print("✅ Found existing precomputed_frame_attrs.csv — skipping generation.")
    

    try:
        frame_attrs_df = pd.read_csv("final_output/precomputed_frame_attrs.csv")
    except FileNotFoundError:
        print("⚠️ Precomputed frame attribute file not found. It will be generated during runtime.")
        frame_attrs_df = pd.DataFrame()  

    # Extract video_id from frame_file (before _frameXYZ.jpg)
    frame_attrs_df["video_id"] = frame_attrs_df["frame_file"].str.extract(r"(?:fake_|real_)?([a-zA-Z0-9_]+(?:_data)?(?:_id\d+_\d+))")
    # Group frame attributes by video_id for fast lookup
    frame_index = frame_attrs_df.groupby("video_id")

    for i, row in enumerate(image_df.itertuples(index=False), 1):
        if st_module and st_module.session_state.get("stop_generation", False):
            print("🛑 Generation manually stopped by user.")
            break
        try:
            image_path = os.path.join("all_data_videos/real_images", row.filename)
            matched_video, best_score = match_best_video_by_face_attributes(
                image_path=image_path,
                frame_df=video_df,
                frame_index=frame_index,
                frame_dir="all_data_frames"
            )

            if not matched_video or not os.path.exists(matched_video):
                print(f"[SKIP] No valid video match for image {row.filename}")
                continue

            #output_filename = f"{row.filename.split('.')[0]}_on_{os.path.basename(matched_video)}"
            image_base = row.filename.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            if image_base.startswith("utkface_real_"):
                image_base = image_base.replace("utkface_real_", "")

            matched_base = os.path.basename(matched_video)
            
            # Simplify matched video filename
            if "celeb_data_real_" in matched_base:
                matched_id = matched_base.replace("celeb_data_real_", "celeb_")
            elif "faceforensics_data_real_" in matched_base:
                matched_id = matched_base.replace("faceforensics_data_real_", "ff_")
            else:
                matched_id = os.path.splitext(matched_base)[0]
                
            output_filename = f"utkface_fake_{image_base}_on_{matched_id}"
            output_path = os.path.join(synthetic_dir, output_filename)

            proc = simswap_single(image_path, matched_video, output_path, st_module=st_module)
            if proc:
                st_module.session_state["simswap_process"] = proc
                proc.wait()
            else:
                continue

            if not os.path.exists(output_path):
                print(f"[ERROR] Failed to generate: {output_filename}")
                continue

            match_log.append({
                "image": row.filename,
                "video": os.path.basename(matched_video),
                "score": best_score  # Make sure to return score from match_best_video_by_face_attributes
            })

            latest_entry = {
                "filename": output_filename,
                "path": output_path,
                "label": "fake",
                "source": "synthetic",
                "age": row.age,
                "age_group": row.age_group
            }
            new_annotations.append(latest_entry)
            
            temp_annopath = "final_output/temp_synthetic_annotations.csv"
            try:
                df_row = pd.DataFrame([latest_entry])
                if os.path.exists(temp_annopath):
                    df_row.to_csv(temp_annopath, mode='a', header=False, index=False)
                else:
                    df_row.to_csv(temp_annopath, index=False)  # write with header if file doesn't exist yet
                print(f"💾 Appended synthetic annotation for: {output_filename}")
            except Exception as e:
                print(f"[ERROR] Could not append temporary synthetic annotation: {e}")

            if streamlit_progress and st_module:
                streamlit_progress.progress(i / total, text=f"Generating ({i}/{total})...")

        except Exception as e:
            print(f"[ERROR] Unexpected failure for row {i}: {e}")
            continue

    