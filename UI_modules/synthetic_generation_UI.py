import streamlit as st
import os
import pandas as pd
from logger import log_action
from synthetic_data_simswap import generate_synthetic_videos
from data_preprocessing import FAKE_BALANCE_TARGET, merge_synthetic_into_balanced_annotations, generate_synthetic_frame_annotations
import json

def render_synthetic_generation_ui():
    

    st.markdown("## 🎭 Generate Synthetic Deepfake Videos")
    
    # Initialize session flags once
    if "generation_started" not in st.session_state:
        st.session_state["generation_started"] = False
    if "stop_generation" not in st.session_state:
        st.session_state["stop_generation"] = False
    if "simswap_process" not in st.session_state:
        st.session_state["simswap_process"] = None

    # Stop button logic — render only if generation has started
    if st.session_state.get("generation_started", False):
        if st.button("🛑 Stop Generation"):
            st.session_state["stop_generation"] = True
            proc = st.session_state.get("simswap_process")
            if proc and proc.poll() is None:
                proc.terminate()
                st.warning("🛑 SimSwap process terminated by user.")

    if st.button("🎬 Prepare & Generate Synthetic Videos"):
        st.session_state["generation_started"] = True
        st.session_state["stop_generation"] = False
            
        try:
            annotations_df = pd.read_csv("final_output/balanced_metadata.csv")
        except FileNotFoundError:
            st.error("❌ No metadata file found (final_output/balanced_metadata.csv). Cannot proceed.")
            return

        real_df = annotations_df[(annotations_df["source"] == "UTKFace") & (annotations_df["label"] == "real")]
        if real_df.empty:
            st.warning("⚠️ No UTKFace real images available for synthetic generation.")
            return

        fake_df = annotations_df[
            (annotations_df["label"] == "fake") & 
            (annotations_df["source"].isin(["celeb", "faceforensics"]))
        ]
        
        try:
            with open("final_output/balance_config.json", "r") as f:
                config = json.load(f)
                FAKE_BALANCE_TARGET = config.get("fake_balance_target")
        except Exception:
            FAKE_BALANCE_TARGET = None
        
        if FAKE_BALANCE_TARGET is None:
            st.error("❌ FAKE_BALANCE_TARGET not set or could not be loaded. Please run the balancing step first.")
            return

        # Prepare plan
        synthetic_plan = {}
        for age_group in sorted(real_df["age_group"].unique()):
            current_fake = fake_df[fake_df["age_group"] == age_group].shape[0]
            required = max(FAKE_BALANCE_TARGET - current_fake, 0)
            
            print(f"[INFO] Age Group: {age_group} | Target: {FAKE_BALANCE_TARGET} | Current Fake: {current_fake} | Synthetic Needed: {required}")

            if required > 0:
                synthetic_plan[age_group] = required

        if not synthetic_plan:
            st.success("✅ Fake videos are already balanced across all age groups.")
            return

        st.markdown("#### 🧠 Synthetic Generation Plan")
        st.dataframe(pd.DataFrame({
            "Age Group": list(synthetic_plan.keys()),
            "Synthetic Videos to Create": list(synthetic_plan.values())
        }))

        selected_rows = []
        for age_group, num_videos in synthetic_plan.items():
            available = real_df[real_df["age_group"] == age_group]
            if not available.empty:
                selected = available.sample(n=min(num_videos, len(available)))
                selected_rows.append(selected)

        if selected_rows:
            final_df = pd.concat(selected_rows).reset_index(drop=True)
            progress_bar = st.progress(0, text="Generating synthetic videos...")
            generate_synthetic_videos(final_df, streamlit_progress=progress_bar, st_module=st)
            frame_rate = st.session_state.get("selected_frame_rate", 30)
            
            merge_synthetic_into_balanced_annotations()
            generate_synthetic_frame_annotations(frame_rate = frame_rate)
            
            progress_bar.empty()
            st.success("✅ Synthetic videos generated and metadata updated.")
            
    