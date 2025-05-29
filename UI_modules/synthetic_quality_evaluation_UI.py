import streamlit as st
import os
import cv2
from synthetic_quality_evaluation import evaluate_synthetic_video_quality
def render_quality_evaluation_ui():
    st.markdown("### 🎥 Synthetic Data Quality Evaluation")

    real_img_dir = "all_data_videos/real_images"
    synth_video_dir = "all_data_videos/synthetic"

    synth_videos = [f for f in os.listdir(synth_video_dir) if f.lower().endswith('.mp4')]

    if not synth_videos:
        st.warning("❗ No synthetic videos found.")
        return

    synth_choice = st.selectbox("Select Synthetic Video:", synth_videos)

    if st.button("🔍 Evaluate Quality"):
        synth_path = os.path.join(synth_video_dir, synth_choice)

        # Extract base name to find corresponding real image
        base_name = synth_choice.split("_on_")[0]
        possible_real_images = [
            f for f in os.listdir(real_img_dir)
            if f.startswith(base_name) and f.lower().endswith(('.jpg', '.png'))
        ]

        if not possible_real_images:
            st.error(f"No matching real image found for: {base_name}")
            return

        real_img_path = os.path.join(real_img_dir, possible_real_images[0])

        try:
            result = evaluate_synthetic_video_quality(real_img_path, synth_path)

            st.success("✅ Evaluation complete!")
            st.write(f"**SSIM Score:** {result['SSIM']:.4f}")
            st.write(f"**PSNR Score:** {result['PSNR']:.2f} dB")

            real_img = cv2.imread(real_img_path)
            synth_frame = cv2.VideoCapture(synth_path).read()[1]

            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            synth_frame = cv2.cvtColor(synth_frame, cv2.COLOR_BGR2RGB)

            st.image([real_img, synth_frame], caption=["Real Image", "Synthetic Frame (Middle)"], width=300)

        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")