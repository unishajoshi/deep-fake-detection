import streamlit as st
import os
import cv2
from synthetic_quality_evaluation import evaluate_synthetic_video_quality, extract_middle_frame

def render_quality_evaluation_ui():
    st.markdown("### 🎥 Synthetic Data Quality Evaluation")

    real_img_dir = "all_data_videos/real_images"
    synth_video_dir = "all_data_videos/synthetic"
    os.makedirs(os.path.dirname(synth_video_dir), exist_ok=True)

    synth_videos = [f for f in os.listdir(synth_video_dir) if f.lower().endswith('.mp4')]

    if not synth_videos:
        st.warning("❗ No synthetic videos found.")
        return

    synth_choice = st.selectbox("🎯 Select Synthetic Video (or evaluate all):", ["(Evaluate All)"] + synth_videos)

    if st.button("🔍 Evaluate Quality"):
        scores = []
        evaluation_results = {}  # For session retention

        target_videos = synth_videos if synth_choice == "(Evaluate All)" else [synth_choice]

        for synth_name in target_videos:
            synth_path = os.path.join(synth_video_dir, synth_name)

            # Extract base name to find corresponding real image
            base_name = synth_name.split("_on_")[0]
            possible_real_images = [
                f for f in os.listdir(real_img_dir)
                if f.startswith(base_name) and f.lower().endswith(('.jpg', '.png'))
            ]

            if not possible_real_images:
                st.warning(f"⚠️ No matching real image found for: `{base_name}`")
                continue

            real_img_path = os.path.join(real_img_dir, possible_real_images[0])

            try:
                result = evaluate_synthetic_video_quality(real_img_path, synth_path)
                scores.append(result)
                evaluation_results[synth_name] = result

                st.markdown(f"#### 📹 `{synth_name}`")
                st.write(f"**SSIM Score:** `{result['SSIM']:.4f}`")
                st.write(f"**PSNR Score:** `{result['PSNR']:.2f} dB`")

                if len(target_videos) == 1:
                    real_img = cv2.imread(real_img_path)
                    synth_frame = extract_middle_frame(synth_path)

                    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                    synth_frame = cv2.cvtColor(synth_frame, cv2.COLOR_BGR2RGB)

                    st.image([real_img, synth_frame], caption=["Real Image", "Synthetic Frame (Middle)"], width=300)

            except Exception as e:
                st.error(f"❌ Error evaluating `{synth_name}`: {str(e)}")

        if len(scores) > 1:
            avg_ssim = sum([s['SSIM'] for s in scores]) / len(scores)
            avg_psnr = sum([s['PSNR'] for s in scores]) / len(scores)

            st.markdown("---")
            st.success("📊 **Average Quality Metrics Across All Synthetic Videos:**")
            st.write(f"**Average SSIM:** `{avg_ssim:.4f}`")
            st.write(f"**Average PSNR:** `{avg_psnr:.2f} dB`")

            # Retain average in session
            evaluation_results['__average__'] = {'SSIM': avg_ssim, 'PSNR': avg_psnr}

        # Store in session state
        st.session_state['quality_evaluation_results'] = evaluation_results
