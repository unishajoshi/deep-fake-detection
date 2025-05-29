import streamlit as st
from model_trainer import train_models
from logger import log_action

def render_model_training_ui():
    st.markdown("### 🧠 Model Training")

    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "training_done" not in st.session_state:
        st.session_state.training_done = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("XceptionNet", value="XceptionNet" in st.session_state.selected_models):
            if "XceptionNet" not in st.session_state.selected_models:
                st.session_state.selected_models.append("XceptionNet")
        else:
            if "XceptionNet" in st.session_state.selected_models:
                st.session_state.selected_models.remove("XceptionNet")
    with col2:
        if st.checkbox("EfficientNet", value="EfficientNet" in st.session_state.selected_models):
            if "EfficientNet" not in st.session_state.selected_models:
                st.session_state.selected_models.append("EfficientNet")
        else:
            if "EfficientNet" in st.session_state.selected_models:
                st.session_state.selected_models.remove("EfficientNet")
    with col3:
        if st.checkbox("LipForensics", value="LipForensics" in st.session_state.selected_models):
            if "LipForensics" not in st.session_state.selected_models:
                st.session_state.selected_models.append("LipForensics")
        else:
            if "LipForensics" in st.session_state.selected_models:
                st.session_state.selected_models.remove("LipForensics")

    if st.button("Train Selected Models"):
        if not st.session_state.selected_models:
            st.warning("⚠️ Please select at least one model.")
            st.session_state.training_done = False
        else:
            with st.spinner("Training..."):
                st.session_state.trained_models = train_models(
                    st.session_state.selected_models,
                    train_csv="train_split.csv",
                    streamlit_mode=True
                )
            st.session_state.training_done = True

    if st.session_state.training_done:
        st.success("✅ Model training complete.")
        log_action("Model Training", "SUCCESS", "Models trained on balanced dataset.")