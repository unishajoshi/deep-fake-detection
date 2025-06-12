import streamlit as st
import asyncio
import os
import psutil

# Project-wide setup
os.makedirs("logs", exist_ok=True)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    
try:
    import os
    os.environ["STREAMLIT_WATCH_MODULE"] = "false"
    print("[INFO] Streamlit module watching disabled.")
except Exception as e:
    print(f"[WARNING] Failed to set STREAMLIT_WATCH_MODULE: {e}")
# UI Modules
from clean_data import clean_videos_and_files
from UI_modules.title_section_UI import render_title_section_ui
from UI_modules.sidebar_disclaimer_UI import render_sidebar_disclaimer_ui
from UI_modules.video_import_UI import render_image_import_ui
from UI_modules.video_import_UI import render_video_upload_ui, render_import_videos_ui
from UI_modules.video_cleaning_UI import render_video_cleaning_ui
from UI_modules.frame_extraction_UI import render_frame_extraction_ui
from UI_modules.frame_preview_UI import render_frame_preview_ui
from UI_modules.age_annotation_UI import render_age_annotation_ui
from UI_modules.visualization_UI import render_visualization_ui
from UI_modules.balance_dataset_UI import render_balance_dataset_ui
from UI_modules.synthetic_generation_UI import render_synthetic_generation_ui
from UI_modules.model_training_UI import render_model_training_ui
from UI_modules.model_evaluation_UI import render_model_evaluation_ui
from UI_modules.source_training_eval_UI import render_source_training_eval_ui
from UI_modules.gradcam_UI import render_gradcam_ui
from UI_modules.report_export_UI import render_report_export_ui
from UI_modules.dataset_export_UI import render_dataset_export_ui
from UI_modules.thank_you_UI import render_thank_you_ui
from UI_modules.synthetic_quality_evaluation_UI import render_quality_evaluation_ui
from UI_modules.train_test_split_UI import render_train_test_split_ui
from UI_modules.cleanup_session_UI import render_session_cleaning_ui



# Initialize session state
st.session_state.setdefault("selected_models", [])
st.session_state.setdefault("trained_models", {})
st.session_state.setdefault("train_df", None)
st.session_state.setdefault("test_df", None)

# Main app flow
def main():
    render_title_section_ui()
    render_sidebar_disclaimer_ui()
    if "initial_cleanup_done" not in st.session_state:
        #clean_videos_and_files()
        st.session_state.initial_cleanup_done = True
        
    render_image_import_ui()   
    render_video_upload_ui()
    render_import_videos_ui()
    render_video_cleaning_ui()
    render_frame_extraction_ui()
    render_frame_preview_ui()
    render_age_annotation_ui()
    render_visualization_ui()
    render_balance_dataset_ui()
    render_synthetic_generation_ui()
    render_quality_evaluation_ui()
    render_train_test_split_ui()
    render_model_training_ui()
    render_model_evaluation_ui()
    render_source_training_eval_ui()
    render_gradcam_ui()
    render_report_export_ui()
    render_dataset_export_ui()
    render_session_cleaning_ui()
    render_thank_you_ui()

if __name__ == "__main__":
    main()
