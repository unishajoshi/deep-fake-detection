import streamlit as st
import pandas as pd
from visualization import get_age_label_source_table, show_age_distribution_pie_charts
from logger import log_action

def render_visualization_ui():
    st.markdown("---")
    st.subheader("📊 Age & Label Distribution by Dataset")

    if "age_dist_table" not in st.session_state:
        st.session_state.age_dist_table = None
    if "show_age_dist_table" not in st.session_state:
        st.session_state.show_age_dist_table = False

    if st.button("📊 Show Distribution Table"):
        pivot = get_age_label_source_table()
        if pivot is not None:
            st.session_state.age_dist_table = pivot
            st.session_state.show_age_dist_table = True
            log_action("Age distribution table", "SUCCESS", "Loaded distribution table.")
        else:
            st.error("❌ `annotations.csv` not found. Please run age annotation first.")
            st.session_state.show_age_dist_table = False
            log_action("Age distribution table", "FAIL", "Failed to load distribution table.")

    if st.session_state.show_age_dist_table and st.session_state.age_dist_table is not None:
        st.dataframe(st.session_state.age_dist_table)

    if "pie_chart_figures" not in st.session_state:
        st.session_state.pie_chart_figures = []
    if "show_pie_charts" not in st.session_state:
        st.session_state.show_pie_charts = False

    if st.button("🥧 Show Age Group Pie Charts"):
        st.session_state.pie_chart_figures = show_age_distribution_pie_charts(return_figures=True)
        st.session_state.show_pie_charts = True

    if st.session_state.show_pie_charts and st.session_state.pie_chart_figures:
        st.markdown("### 🥧 Age Group Distribution by Source and Label")
        
        cols = st.columns(3)
        for i, (title, fig) in enumerate(st.session_state.pie_chart_figures):
            with cols[i % 3]:
                st.markdown(f"#### {title}")
                st.pyplot(fig)