import streamlit as st
import os
import pandas as pd
import cv2
from synthetic_quality_evaluation import evaluate_synthetic_video_quality, extract_middle_frame

def render_quality_evaluation_ui():
    import streamlit as st
import os
import pandas as pd
import cv2
from synthetic_quality_evaluation import evaluate_synthetic_video_quality, extract_middle_frame

def render_quality_evaluation_ui():
    st.markdown("### 📊 Average Quality Scores (All Synthetic Videos)")

    real_img_dir = "all_data_videos/real_images"
    synth_video_dir = "all_data_videos/synthetic"
    os.makedirs(synth_video_dir, exist_ok=True)

    synth_videos = [f for f in os.listdir(synth_video_dir) if f.lower().endswith('.mp4')]
    
    # Trigger button
    if st.button("🧮 Calculate Average SSIM & PSNR"):
        avg_scores = []
        for synth_name in synth_videos:
            synth_path = os.path.join(synth_video_dir, synth_name)
            base_name = synth_name.split("_on_")[0].replace("fake", "real")
            possible_real_images = [
                f for f in os.listdir(real_img_dir)
                if f.startswith(base_name) and f.lower().endswith(('.jpg', '.png'))
            ]
            if not possible_real_images:
                continue

            real_img_path = os.path.join(real_img_dir, possible_real_images[0])
            try:
                result = evaluate_synthetic_video_quality(real_img_path, synth_path)
                avg_scores.append(result)
            except:
                continue

        st.session_state.avg_quality_scores = avg_scores
        st.success("✅ Average quality calculated and cached for this session.")

    # Show cached results if available
    if "avg_quality_scores" in st.session_state:
        avg_scores = st.session_state.avg_quality_scores
        if avg_scores:
            mean_ssim = sum([s['SSIM'] for s in avg_scores]) / len(avg_scores)
            mean_psnr = sum([s['PSNR'] for s in avg_scores]) / len(avg_scores)

            st.write(f"**Average SSIM:** `{mean_ssim:.4f}`")
            st.write(f"**Average PSNR:** `{mean_psnr:.2f} dB`")
        else:
            st.warning("⚠️ No valid synthetic-real image pairs were found.")
    else:
        st.info("🔔 Click the button above to calculate average quality scores.")

    # --- Single video evaluation section ---
    st.markdown("---")
    st.markdown("### 🎥 Synthetic Data Quality Evaluation (Individual Video)")

    if not synth_videos:
        st.warning("❗ No synthetic videos found.")
        return

    synth_choice = st.selectbox("🎯 Select a Synthetic Video to Evaluate:", synth_videos)

    if st.button("🔍 Evaluate Selected Video"):
        synth_path = os.path.join(synth_video_dir, synth_choice)
        base_name = synth_choice.split("_on_")[0].replace("fake", "real")
        possible_real_images = [
            f for f in os.listdir(real_img_dir)
            if f.startswith(base_name) and f.lower().endswith(('.jpg', '.png'))
        ]

        if not possible_real_images:
            st.warning(f"⚠️ No matching real image found for: `{base_name}`")
        else:
            real_img_path = os.path.join(real_img_dir, possible_real_images[0])
            try:
                result = evaluate_synthetic_video_quality(real_img_path, synth_path)

                st.markdown(f"#### 📹 `{synth_choice}`")
                st.write(f"**SSIM Score:** `{result['SSIM']:.4f}`")
                st.write(f"**PSNR Score:** `{result['PSNR']:.2f} dB`")

                real_img = cv2.imread(real_img_path)
                synth_frame = extract_middle_frame(synth_path)

                real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                synth_frame = cv2.cvtColor(synth_frame, cv2.COLOR_BGR2RGB)

                st.image([real_img, synth_frame], caption=["Real Image", "Synthetic Frame (Middle)"], width=300)
            except Exception as e:
                st.error(f"❌ Error evaluating `{synth_choice}`: {str(e)}")
                
    st.markdown("---")
    try:
        final_metadata = pd.read_csv("final_output/balanced_annotations.csv")
        summary = final_metadata.groupby(["age_group", "label"]).size().unstack(fill_value=0)
        st.markdown("### 📊 Age-balanced Dataset Prepared for Training")
        st.dataframe(summary)
    except Exception as e:
        st.error(f"⚠️ Failed to load final metadata: {e}")
