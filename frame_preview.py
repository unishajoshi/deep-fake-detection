#------------------ Section for: DISPLAY SAMPLE FRAMES--------------

import os
import cv2
import random
import streamlit as st

def display_grid_pair(real_frames, fake_frames):

    col1, col2 = st.columns(2)

    def show_grid(frames, label, container,border_color):
        with container:
            container.markdown(f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; margin-bottom: 20px;">
                    <h4 style='text-align: center;'>{label}</h4>
                """,
                unsafe_allow_html=True
            )
            rows, cols = 3, 3
            square_size = 150
            for i in range(rows):
                columns = st.columns(cols)
                for j in range(cols):
                    index = i * cols + j
                    if index < len(frames):
                        img, _ = frames[index]
                        img_resized = cv2.resize(img, (square_size, square_size))
                        columns[j].image(img_resized, channels="BGR", width=square_size)
                    else:
                        columns[j].empty()

    show_grid(real_frames, "🟢 Real", col1, "#4CAF50")
    show_grid(fake_frames, "🔴 Fake", col2, "#F44336")

def preview_sample_frames(real_dir, fake_dir, return_images=False, frame_output_dir="sample_preview_frames", frame_rate=10, count=9):
    os.makedirs(frame_output_dir, exist_ok=True)

    def get_sample_frames(video_dir, label):
        sample_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
        sample_videos = random.sample(sample_videos, min(count, len(sample_videos)))
        preview_frames = []

        for video in sample_videos:
            video_path = os.path.join(video_dir, video)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_rate)
            success, frame = cap.read()
            cap.release()
            if success:
                preview_frames.append((frame, f"{label}: {video}"))  # keep frame as BGR np.array

        return preview_frames

    real_frames = get_sample_frames(real_dir, "Real")
    fake_frames = get_sample_frames(fake_dir, "Fake")

    if return_images:
        return real_frames, fake_frames  # ✅ raw BGR frames with captions
    else:
        display_grid_pair(real_frames, fake_frames)