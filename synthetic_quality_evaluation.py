import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from math import log10
import os

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Failed to read middle frame from video")
    return frame

def evaluate_synthetic_video_quality(real_img_path, synthetic_video_path):
    source_img = cv2.imread(real_img_path)
    synth_frame = extract_middle_frame(synthetic_video_path)

    if source_img.shape != synth_frame.shape:
        synth_frame = cv2.resize(synth_frame, (source_img.shape[1], source_img.shape[0]))

    return {
        "PSNR": compute_psnr(source_img, synth_frame),
        "SSIM": compute_ssim(source_img, synth_frame)
    }