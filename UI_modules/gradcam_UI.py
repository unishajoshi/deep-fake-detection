import os
import streamlit as st
import torch.nn as nn
import pandas as pd
from grad_cam import apply_gradcam
from model_trainer import get_model

def render_gradcam_ui():
    st.markdown("---")
    st.subheader("🎯 Age-Specific Deepfake Explanation (Grad-CAM)")

    def get_last_conv_layer(model):
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        return None

    def display_images_in_grid(image_list, captions, title, columns=4):
        st.markdown(f"#### {title}")
        for i in range(0, len(image_list), columns):
            cols = st.columns(columns)
            for j in range(columns):
                idx = i + j
                if idx < len(image_list):
                    cols[j].image(image_list[idx], caption=captions[idx], use_container_width=True)

    selected_model = st.selectbox("🤖 Choose a Model", ["XceptionNet", "EfficientNet", "LipForensics"])

    if st.button("🎯 Run Grad-CAM"):
        df = pd.read_csv("final_output/frame_level_annotations.csv")
        df["video_id"] = df["frame"].apply(lambda x: os.path.splitext(x)[0])
        real_videos = df[df["label"] == "real"].drop_duplicates("video_id")
        fake_videos = df[df["label"] == "fake"].drop_duplicates("video_id")

        real_sample = real_videos.sample(n=min(4, len(real_videos)), random_state=1)
        fake_sample = fake_videos.sample(n=min(4, len(fake_videos)), random_state=2)

        combined_df = pd.concat([real_sample, fake_sample])
        frame_paths = combined_df["path"].tolist()
        labels = combined_df["label"].tolist()

        model = get_model(selected_model)
        target_layer = get_last_conv_layer(model)

        real_images, fake_images = [], []
        real_captions, fake_captions = [], []

        for path, label in zip(frame_paths, labels):
            if not os.path.exists(path): continue
            cam_result = apply_gradcam(model, path, target_layer)
            if label == "real":
                real_images.append(cam_result)
                real_captions.append(os.path.basename(path))
            else:
                fake_images.append(cam_result)
                fake_captions.append(os.path.basename(path))

        display_images_in_grid(real_images, real_captions, title="🟢 Real Samples")
        display_images_in_grid(fake_images, fake_captions, title="🔴 Fake Samples")
        
        st.session_state.gradcam_images_real = real_images
        st.session_state.gradcam_images_fake = fake_images
        st.session_state.gradcam_captions_real = real_captions
        st.session_state.gradcam_captions_fake = fake_captions