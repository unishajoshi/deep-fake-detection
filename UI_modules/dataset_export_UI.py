import os
import streamlit as st
from io import BytesIO
import pandas as pd
import zipfile

def render_dataset_export_ui():
    st.subheader("📦 Download Age-Diverse Data List & Results")

    if st.button("📥 Download Details"):
        output_dir = "final_output"
        zip_buffer = BytesIO()
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
        summary_csv_path = os.path.join(output_dir, "video_index.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for foldername, _, filenames in os.walk(output_dir):
                for filename in filenames:
                    if filename.endswith((".csv", ".pdf")):
                        path = os.path.join(foldername, filename)
                        zipf.write(path, arcname=os.path.relpath(path, start=output_dir))

        zip_buffer.seek(0)
        st.download_button("⬇️ Download ZIP Archive", zip_buffer, "age_diverse_dataset_outputs.zip", mime="application/zip")