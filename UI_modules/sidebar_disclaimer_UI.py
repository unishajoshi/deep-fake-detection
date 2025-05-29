import streamlit as st

def render_sidebar_disclaimer_ui():
    with st.sidebar.expander("❓ Help & Instructions", expanded=False):
        st.markdown("""
        ### 🛠️ Application Functionality
        This app builds and evaluates an **age-diverse deepfake detection pipeline**:
        1. 📤 Import Videos (celeb-DF, FaceForensics++)
        2. 🧹 Preprocessing
        3. 🖼️ Frame Extraction
        4. 🔍 Frame Preview
        5. 🧓 Age Annotation
        6. 📊 Visualization
        7. ⚖️ Dataset Balancing
        8. 🔄 Train-Test Split
        9. 🧠 Model Training
        10. 📊 Evaluation
        11. 🌀 SimSwap Deepfake Generation
        12. 🎯 Grad-CAM
        13. 📄 PDF Reporting
        14. 📦 Dataset Export
        """)

    with st.sidebar.expander("🔐 Security Disclaimer", expanded=False):
        st.markdown("""
        - Public datasets only: celeb-DF-v2, FaceForensics++
        - Local processing, no video files shared externally
        - Educational/research purposes only
        """)

    with st.sidebar.expander("🔐 Data Availability Disclaimer", expanded=False):
        st.markdown("""
        - Raw datasets are NOT downloadable through this tool
        - Metadata (paths, labels, age groups) is exportable
        - Researchers can re-request datasets from:
            - [celeb-DF-v2](https://github.com/DigitalTrustLab/celeb-DF)
            - [FaceForensics++](https://github.com/ondyari/FaceForensics)
        """)