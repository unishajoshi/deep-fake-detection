import os
import pandas as pd
from data_preprocessing import balance_dataset, export_balanced_dataset, generate_frame_level_annotations
from visualization import get_age_label_source_table
import streamlit as st

def balance_and_export_dataset(method="oversample"):
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

    if not os.path.exists("annotations.csv"):
        result["messages"].append("❌ annotations.csv not found. Please run age annotation first.")
        return result

    if os.path.exists("balanced_annotations.csv"):
        os.remove("balanced_annotations.csv")
    if os.path.exists("frame_level_annotations.csv"):
        os.remove("frame_level_annotations.csv")

    df = pd.read_csv("annotations.csv")
    result["pre_distribution"] = get_age_label_source_table("annotations.csv")

    # Balance
    df_balanced = balance_dataset("annotations.csv", method=method)
    df_balanced.to_csv("balanced_annotations.csv", index=False)
    result["post_distribution"] = get_age_label_source_table("balanced_annotations.csv")

    # Generate frame-level annotations
    generate_frame_level_annotations(mode="balanced")
    result["frame_annotated"] = True

    # Export balanced videos
    copied, missing, out_path = export_balanced_dataset("balanced_annotations.csv")
    result["copied"] = copied
    result["missing"] = missing
    result["export_path"] = out_path
    result["status"] = "success"
    return result


