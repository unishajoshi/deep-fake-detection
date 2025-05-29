import os
import streamlit as st
import pandas as pd
from model_trainer import train_models_on_source
from evaluation import evaluate_on_all_sets_for_trained_models, flatten_results_grouped

def render_source_training_eval_ui():
    st.markdown("## 🔄 Deepfake Detection with Original Data")
    st.markdown("### 🏋️ Source-Specific Model Training")

    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "source_training_done" not in st.session_state:
        st.session_state.source_training_done = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("XceptionNet", value="XceptionNet" in st.session_state.selected_models, key="src_xcep"):
            st.session_state.selected_models.append("XceptionNet")
    with col2:
        if st.checkbox("EfficientNet", value="EfficientNet" in st.session_state.selected_models, key="src_eff"):
            st.session_state.selected_models.append("EfficientNet")
    with col3:
        if st.checkbox("LipForensics", value="LipForensics" in st.session_state.selected_models, key="src_lip"):
            st.session_state.selected_models.append("LipForensics")

    train_source = st.radio("Select training dataset source:", ["celeb", "FaceForensics++"], horizontal=True)

    if st.button("Train Models"):
        with st.spinner(f"Training on {train_source}..."):
            trained_models, test_df = train_models_on_source(
                source_name=train_source.lower(),
                metadata_csv="frame_level_annotations_original.csv",
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
                    streamlit_mode=True
                )
                df_cross_eval = flatten_results_grouped(cross_results)
                st.session_state.cross_eval_df = df_cross_eval
                st.session_state.cross_eval_done = True
                output_path = f"final_output/{train_source.lower()}_cross_eval.csv"
                df_cross_eval.to_csv(output_path, index=False)
                st.success("✅ Cross-evaluation complete.")
                st.dataframe(df_cross_eval, use_container_width=True)