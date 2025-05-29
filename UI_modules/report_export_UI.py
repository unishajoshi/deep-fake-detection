import os
import streamlit as st
import pandas as pd
from wrapup import create_full_pdf_report

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
        sample_frame_paths = []
        for root, _, files in os.walk("all_data_frames"):
            for f in files:
                if f.endswith(".jpg"):
                    sample_frame_paths.append(os.path.join(root, f))
        sample_frame_paths = sample_frame_paths[:6]

        age_annotation_summary = {}
        if os.path.exists("annotations.csv"):
            age_df = pd.read_csv("annotations.csv")
            age_annotation_summary = age_df["age_group"].value_counts().to_dict()

        balance_summary = {}
        if os.path.exists("balanced_metadata.csv"):
            balanced_df = pd.read_csv("balanced_metadata.csv")
            balance_summary = balanced_df["age_group"].value_counts().to_dict()

        eval_balanced = pd.read_csv("final_output/evaluation_results.csv") if os.path.exists("final_output/evaluation_results.csv") else None
        eval_agewise = pd.read_csv("final_output/age_specific_evaluation.csv") if os.path.exists("final_output/age_specific_evaluation.csv") else None
        eval_celeb = pd.read_csv("final_output/celeb_cross_eval.csv") if os.path.exists("final_output/celeb_cross_eval.csv") else None
        eval_ffpp = pd.read_csv("final_output/faceforensics_cross_eval.csv") if os.path.exists("final_output/faceforensics_cross_eval.csv") else None

        gradcam_paths = []
        for f in os.listdir("temp_uploads") if os.path.exists("temp_uploads") else []:
            if f.endswith((".jpg", ".png")):
                gradcam_paths.append(os.path.join("temp_uploads", f))

        report_path = create_full_pdf_report(
            output_path="final_output/final_report.pdf",
            dataset_summary=dataset_summary,
            cleaning_notes="- Low-res files removed.",
            frame_count_summary=frame_count_summary,
            sample_frame_paths=sample_frame_paths,
            age_annotation_summary=age_annotation_summary,
            balance_summary=balance_summary,
            eval_balanced=eval_balanced,
            eval_agewise=eval_agewise,
            eval_celeb=eval_celeb,
            eval_ffpp=eval_ffpp,
            gradcam_paths=gradcam_paths
        )

        with open(report_path, "rb") as f:
            st.download_button("⬇️ Download Project Summary Report", f, "Deepfake_Project_Report.pdf", mime="application/pdf")