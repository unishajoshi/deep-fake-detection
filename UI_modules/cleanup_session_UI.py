import os
import streamlit as st
from clean_data import clean_videos_and_files, remove_pdf_files



# ------------------------------
# 🔁 Reset and Cleanup Section
# ------------------------------

def render_session_cleaning_ui():
    st.markdown("---")
    st.subheader("♻️ Cleanup & Reset")
    
    if st.button("🔄 Reset Session and Delete All Files"):
        with st.spinner("Cleaning up..."):
            clean_videos_and_files()       
            remove_pdf_files()             
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("✅ Session and all files successfully removed.")
        st.query_params.update({"reset": "true"})
        st.rerun()
        log_action("Cleanup", "SUCCESS", "Removed all generated files and PDFs.")
    
    st.info("🔒 **Reminder:** Use the Reset Session button to clear memory and files before exiting.")