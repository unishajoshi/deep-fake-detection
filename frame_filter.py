# frame_filter_mtcnn.py

import os
import cv2
import torch
import numpy as np
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor, as_completed
from facenet_pytorch import MTCNN

# -------------------
# CONFIGURATION
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# Thresholds
BLUR_THRESHOLD = 60
DARK_THRESHOLD = 30
BRIGHT_THRESHOLD = 235
MIN_RESOLUTION = (64, 64)
MIN_FACE_SIZE = 10  # pixels
MIN_CONFIDENCE = 0.90
MAX_YAW_DEG = 45
BORDER_MARGIN = 5

# -------------------
# IMAGE QUALITY CHECKS
# -------------------
def is_low_light(img):
    return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < DARK_THRESHOLD

def is_over_exposed(img):
    return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > BRIGHT_THRESHOLD

def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

# -------------------
# FACE VALIDITY CHECK (MTCNN + LANDMARKS)
# -------------------
def is_valid_face_mtcnn(img):
    h, w = img.shape[:2]
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

    if boxes is None or len(boxes) == 0 or probs[0] < MIN_CONFIDENCE:
        return False  # No face or low confidence

    x1, y1, x2, y2 = boxes[0]
    face_w, face_h = x2 - x1, y2 - y1

    #if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
      #  return False  # Face too small

    if x1 < BORDER_MARGIN or y1 < BORDER_MARGIN or x2 > (w - BORDER_MARGIN) or y2 > (h - BORDER_MARGIN):
        return False  # Partial face near edge

    if landmarks is not None:
        left_eye, right_eye = landmarks[0][0], landmarks[0][1]
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        yaw_angle = abs(np.degrees(np.arctan2(dy, dx)))
        if yaw_angle > MAX_YAW_DEG:
            return False  # Extreme angle

    return True

# -------------------
# FRAME FILTERING LOGIC
# -------------------
def filter_single_frame_mtcnn(file_path, output_dir, hash_set):
    img = cv2.imread(file_path)
    if img is None or img.shape[0] < MIN_RESOLUTION[1] or img.shape[1] < MIN_RESOLUTION[0]:
        return

    if is_low_light(img) or is_over_exposed(img) or is_blurry(img) or not is_valid_face_mtcnn(img):
        return

    img_hash = md5(img.tobytes()).hexdigest()
    if img_hash in hash_set:
        return
    hash_set.add(img_hash)

    out_path = os.path.join(output_dir, os.path.basename(file_path))
    cv2.imwrite(out_path, img)

def filter_frames_mtcnn_parallel(input_dir, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
    seen_hashes = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(filter_single_frame_mtcnn, f, output_dir, seen_hashes) for f in frame_paths]
        for _ in as_completed(futures):
            pass

    print(f"✅ Filtered frames saved to: {output_dir}")

def is_valid_frame(img):
    return (
        img is not None and
        img.shape[0] >= MIN_RESOLUTION[1] and
        img.shape[1] >= MIN_RESOLUTION[0] and
        not is_low_light(img) and
        not is_over_exposed(img) and
        not is_blurry(img) and
        is_valid_face_mtcnn(img)
    )
