import os
import streamlit as st
from io import BytesIO
import pandas as pd
import re

def render_dataset_export_ui():
    st.subheader("📦 Download Age-Diverse Data List & Results")
    st.markdown("""
    
    This section allow to **generate and download a structured CSV file** listing age-balanced real and fake videos used in the project.
    This dataset is age-diverse and aims to improve  age fairness in deepfake detection
    The CSV includes:
    - 🏷️ Metadata Attributes: Filename(matching source), Label (real or fake), Source and Age-group
    - 📁 Source dataset (Celeb-DF or FaceForensics++ or UTKFace)
    - 📁 Includes synthetic video created with the combination of UTKFace and one of other video source)
    """)
   
    if st.button("📥 Generate Age-balanced Metadata"):
       try:
           # Load frame-level annotations
            df = pd.read_csv("final_output/frame_level_annotations.csv")
            
            def clean_filename(raw_name):
                if pd.isna(raw_name):
                    return None
                # Step 1: Remove everything up to and including the last "real_" or "fake_"
                trimmed = re.sub(r".*?(real_|fake_)", "", raw_name)
                # Step 2: Remove trailing _frameXXX (e.g., _frame20, _frame300)
                cleaned = re.sub(r"_frame\d+", "", trimmed)
                return cleaned
    
            # Extract base video ID from frame column (e.g., abc_frame120.jpg → abc)
            df["frame"] = df["frame"].apply(clean_filename)
    
            # Use the first row per video_id
            video_df = df.drop_duplicates("frame")[
                ["frame", "label", "source", "age_group"]
            ].copy()
    
            # Rename to match expected output column name
            video_df.rename(columns={"frame": "filename"}, inplace=True)
    
            # Preview in Streamlit
            st.dataframe(video_df)
    
            # Convert to CSV and make downloadable
            csv_data = video_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Metadata File",
                data=csv_data,
                file_name="video_index.csv",
                mime="text/csv"
            )
       except Exception as e:
            st.error(f"❌ Failed to generate video index: {e}")