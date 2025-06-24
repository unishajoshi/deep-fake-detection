import os
import streamlit as st
import pandas as pd
from model_trainer import train_models_on_source
from evaluation import evaluate_on_all_sets_for_trained_models, flatten_results_grouped

def render_source_training_eval_ui():
    st.markdown("## 🔄 Deepfake Detection with Original Data")
    st.markdown("### 🏋️ Source-Specific Model Training")
    
    # Initialize only once
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    
    if "source_training_done" not in st.session_state:
        st.session_state.source_training_done = False
    
    # Temporary checkbox states
    xcep = st.checkbox("XceptionNet", key="src_xcep")
    eff = st.checkbox("EfficientNet", key="src_eff")
    lip = st.checkbox("LipForensics", key="src_lip")
    
    # Overwrite the list completely (no duplicates)
    st.session_state.selected_models = []
    if xcep:
        st.session_state.selected_models.append("XceptionNet")
    if eff:
        st.session_state.selected_models.append("EfficientNet")
    if lip:
        st.session_state.selected_models.append("LipForensics")

    train_source = st.radio("Select training dataset source:", ["celeb", "faceforensics"], horizontal=True)

    if st.button("Train Models"):
        with st.spinner(f"Training on {train_source}..."):
            trained_models, test_df = train_models_on_source(
                source_name=train_source.lower(),
                metadata_csv="final_output/frame_level_annotations_source.csv",
                selected_models=st.session_state.selected_models,
                streamlit_mode=True
            )
        st.session_state.trained_models = trained_models
        st.session_state.source_training_done = True

    if st.session_state.source_training_done:
        st.success("✅ Source-based model training complete.")

    st.markdown("### 🧪 Evaluate Original Source Performance")
    if st.button("Evaluate"):
        if not st.session_state.trained_models:
            st.warning("⚠️ Please train models first.")
        else:
            with st.spinner("Evaluating..."):
                cross_results = evaluate_on_all_sets_for_trained_models(
                    st.session_state.trained_models,
                    train_source.lower(),  # pass the selected source name
                    streamlit_mode=True
                )
                df_cross_eval = flatten_results_grouped(cross_results)
                st.session_state.cross_eval_df = df_cross_eval
                st.session_state.cross_eval_done = True
                output_path = f"final_output/{train_source.lower()}_cross_eval.csv"
                df_cross_eval.to_csv(output_path, index=False)
                st.success("✅ Cross-evaluation complete.")
                st.dataframe(df_cross_eval, use_container_width=True)