import cv2
import os

image_path = "all_data_videos/real_images/utkface_real_1_1_0_20170109190457644.jpg"

print("[TEST] Checking if file exists:", os.path.exists(image_path))

img = cv2.imread(image_path)
if img is None:
    print("[FAIL] OpenCV failed to read image.")
else:
    print("[PASS] Image shape:", img.shape)
