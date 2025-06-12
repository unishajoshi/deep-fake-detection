import os
import streamlit as st
from io import BytesIO
import pandas as pd

def render_dataset_export_ui():
    st.subheader("📦 Download Age-Diverse Data List & Results")
    st.markdown("""
    
    This section allow to **generate and download a structured CSV file** listing balanced real and fake videos used in the project.
    This dataset is age-diverse and aims to improve  age fairness in deepfake detection
    The CSV includes:
    - ✅ Filename (with prefixes removed)
    - 🏷️ Label (real or fake)
    - 📁 Source dataset (Celeb-DF or FaceForensics++ or UTKFace)
    """)
    if st.button("📥 Generate Video Index"):
        video_summary = []

        for label in ["real", "fake"]:
            folder = os.path.join("all_data_videos", label)
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    if filename.endswith((".mp4", ".avi")):
                        source = "celeb" if "celeb" in filename.lower() else "faceforensics"
                        truncated = filename
                        for prefix in ["celeb_data_real_", "celeb_data_fake_", "faceforensics_data_real_", "faceforensics_data_fake_"]:
                            if filename.startswith(prefix):
                                truncated = filename.replace(prefix, "", 1)
                                break
                        video_summary.append({"filename": truncated, "label": label, "source": source})

        summary_df = pd.DataFrame(video_summary)

        # Show preview in app
        st.dataframe(summary_df)

        # Convert to CSV and make downloadable
        csv_data = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Video Index CSV",
            data=csv_data,
            file_name="video_index.csv",
            mime="text/csv"
        )