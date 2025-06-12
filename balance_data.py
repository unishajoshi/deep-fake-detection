import os
import pandas as pd
from data_preprocessing import balance_dataset, export_balanced_dataset, generate_frame_level_annotations

from visualization import get_age_label_source_table
import streamlit as st

def balance_and_export_dataset():
    result = {
        "pre_distribution": None,
        "post_distribution": None,
        "copied": 0,
        "missing": [],
        "export_path": None,
        "frame_annotated": False,
        "status": "error",
        "messages": [],
    }

    if not os.path.exists("all_data_videos/annotations.csv"):
        result["messages"].append("❌ annotations.csv not found. Please run age annotation first.")
        return result

    if os.path.exists("final_output/balanced_annotations.csv"):
        os.remove("final_output/balanced_annotations.csv")
    if os.path.exists("final_output/frame_level_annotations.csv"):
        os.remove("final_output/frame_level_annotations.csv")

    df = pd.read_csv("all_data_videos/annotations.csv")
    result["pre_distribution"] = get_age_label_source_table("all_data_videos/annotations.csv")

    # Balance
    try:
        df_balanced = balance_dataset("all_data_videos/annotations.csv")
    except ValueError as ve:
        result["messages"].append(str(ve))
        result["status"] = "error"
        return result
        
    df_balanced.to_csv("final_output/balanced_annotations.csv", index=False)
    result["post_distribution"] = get_age_label_source_table("final_output/balanced_annotations.csv")

    # Generate frame-level annotations
    generate_frame_level_annotations(mode="balanced")
    result["frame_annotated"] = True

    # Export balanced videos
    copied, missing, out_path = export_balanced_dataset("final_output/balanced_annotations.csv")
    result["copied"] = copied
    result["missing"] = missing
    result["export_path"] = out_path
    result["status"] = "success"
    return result


