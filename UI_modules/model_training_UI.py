import streamlit as st
from model_trainer import train_models
from logger import log_action

def render_model_training_ui():
    st.markdown("### 🧠 Model Training")

    # Initialize session state variables
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    if "training_done" not in st.session_state:
        st.session_state.training_done = False

    # Use checkboxes to select models
    st.markdown("Select models to train:")

    model_options = ["XceptionNet", "EfficientNet", "LipForensics"]
    selected = []

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("XceptionNet", value="XceptionNet" in st.session_state.selected_models):
            selected.append("XceptionNet")
    with col2:
        if st.checkbox("EfficientNet", value="EfficientNet" in st.session_state.selected_models):
            selected.append("EfficientNet")
    with col3:
        if st.checkbox("LipForensics", value="LipForensics" in st.session_state.selected_models):
            selected.append("LipForensics")

    st.session_state.selected_models = selected

    if st.button("Train Selected Models"):
        if not selected:
            st.warning("⚠️ Please select at least one model.")
            st.session_state.training_done = False
        else:
            with st.spinner("Training..."):
                st.session_state.trained_models = train_models(
                    selected,
                    train_csv="final_output/train_split.csv",
                    streamlit_mode=True
                )
            st.session_state.training_done = True

    if st.session_state.training_done:
        st.success("✅ Model training complete.")
        log_action("Model Training", "SUCCESS", f"Trained: {', '.join(selected)}")