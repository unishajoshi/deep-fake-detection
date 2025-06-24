import os
import streamlit as st
import pandas as pd
from grad_cam import apply_gradcam
from model_trainer import get_model

def render_gradcam_ui():
    st.markdown("---")
    st.subheader("🎯 Age-Specific Deepfake Explanation (Grad-CAM)")

    def display_images_in_grid(image_list, title, columns=4):
        st.markdown(f"#### {title}")
        for i in range(0, len(image_list), columns):
            cols = st.columns(columns)
            for j in range(columns):
                idx = i + j
                if idx < len(image_list):
                    cols[j].image(image_list[idx], use_container_width=True)

    selected_model = st.selectbox("🤖 Choose a Model", ["XceptionNet", "EfficientNet", "LipForensics"])
    checkpoint_path = f"checkpoints/{selected_model}_best.pth"

    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Trained model checkpoint not found: {checkpoint_path}")
        return

    if "gradcam_cache" not in st.session_state:
        st.session_state.gradcam_cache = {}

    run_gradcam = st.button("🎯 Run Grad-CAM")

    # Check for cache first
    cached_result = st.session_state.gradcam_cache.get(selected_model)

    if run_gradcam or not cached_result:
        try:
            df = pd.read_csv("final_output/test_split.csv")
        except Exception as e:
            st.error(f"❌ Could not load test_split.csv: {e}")
            return

        df["video_id"] = df["frame"].apply(lambda x: os.path.splitext(x)[0])
        real_videos = df[df["label"] == "real"].drop_duplicates("video_id")
        fake_videos = df[df["label"] == "fake"].drop_duplicates("video_id")

        real_sample = real_videos.sample(n=min(4, len(real_videos)), random_state=1)
        fake_sample = fake_videos.sample(n=min(4, len(fake_videos)), random_state=2)
        combined_df = pd.concat([real_sample, fake_sample])

        frame_paths = combined_df["path"].tolist()
        labels = combined_df["label"].tolist()

        real_images, fake_images = [], []

        for path, label in zip(frame_paths, labels):
            if not os.path.exists(path):
                continue
            try:
                cam_result = apply_gradcam(
                    model_name=selected_model,
                    img_path=path,
                    model_loader=get_model
                )
                if label == "real":
                    real_images.append(cam_result)
                else:
                    fake_images.append(cam_result)
            except Exception as e:
                st.warning(f"⚠️ Grad-CAM failed for {path}: {e}")

        # Cache it
        st.session_state.gradcam_cache[selected_model] = {
            "real_images": real_images,
            "fake_images": fake_images,
        }

        # Store for PDF/report
        st.session_state.gradcam_images_real = real_images
        st.session_state.gradcam_images_fake = fake_images

        display_images_in_grid(real_images, title="🟢 Real Samples")
        display_images_in_grid(fake_images, title="🔴 Fake Samples")

    elif cached_result:
        display_images_in_grid(cached_result["real_images"], title="🟢 Real Samples")
        display_images_in_grid(cached_result["fake_images"], title="🔴 Fake Samples")
