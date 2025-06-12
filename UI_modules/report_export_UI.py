import os
import streamlit as st
import pandas as pd
from pdf_report_generation import create_full_pdf_report

def render_report_export_ui():
    st.subheader("📄 Download Project Summary Report")

    if st.button("📝 Generate PDF Report"):
        dataset_summary = {
            "celeb_real": len([f for f in os.listdir("all_data_videos/real") if "celeb" in f.lower()]),
            "celeb_fake": len([f for f in os.listdir("all_data_videos/fake") if "celeb" in f.lower()]),
            "ffpp_real": len([f for f in os.listdir("all_data_videos/real") if "faceforensics" in f.lower()]),
            "ffpp_fake": len([f for f in os.listdir("all_data_videos/fake") if "faceforensics" in f.lower()])
        }

        frame_count_summary = len(os.listdir("all_data_frames")) if os.path.exists("all_data_frames") else 0
        real_frames = []
        fake_frames = []
        real_video_ids = set()
        fake_video_ids = set()
        for root, _, files in os.walk("all_data_frames"):
            for f in files:
                if f.endswith(".jpg"):
                    file_path = os.path.join(root, f)

                    # Extract video identifier from filename (before '_frame')
                    # Example: "real_video123_frame015.jpg" → "real_video123"
                    video_id = f.rsplit('_frame', 1)[0].lower()

                    if "real" in f.lower() and video_id not in real_video_ids and len(real_frames) < 3:
                        real_frames.append(file_path)
                        real_video_ids.add(video_id)

                    elif "fake" in f.lower() and video_id not in fake_video_ids and len(fake_frames) < 3:
                        fake_frames.append(file_path)
                        fake_video_ids.add(video_id)

                # Break early if enough samples collected
                if len(real_frames) == 3 and len(fake_frames) == 3:
                    break
            if len(real_frames) == 3 and len(fake_frames) == 3:
                break
                
        age_annotation_summary = {}
        if os.path.exists("all_data_videos/annotations.csv"):
            age_df = pd.read_csv("all_data_videos/annotations.csv")
            age_annotation_summary = age_df["age_group"].value_counts().to_dict()

        balance_summary = {}
        if os.path.exists("final_output/balanced_metadata.csv"):
            balanced_df = pd.read_csv("final_output/balanced_metadata.csv")
            balance_summary = balanced_df["age_group"].value_counts().to_dict()
       
        gradcam_paths = []
        for f in os.listdir("temp_uploads") if os.path.exists("temp_uploads") else []:
            if f.endswith((".jpg", ".png")):
                gradcam_paths.append(os.path.join("temp_uploads", f))

        report_path = create_full_pdf_report(
            output_path="final_output/final_report.pdf",
            dataset_summary=dataset_summary,
            cleaning_notes="- Low-res files removed.",
            frame_count_summary=frame_count_summary,
            real_frames=real_frames,
            fake_frames=fake_frames,
            age_annotation_summary=age_annotation_summary,
            balance_summary=balance_summary,
            gradcam_paths=gradcam_paths
        )

        with open(report_path, "rb") as f:
            st.download_button("⬇️ Download Project Summary Report", f, "Deepfake_Project_Report.pdf", mime="application/pdf")