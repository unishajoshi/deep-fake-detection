import streamlit as st
import os
import pandas as pd
from logger import log_action
from synthetic_data_simswap import generate_synthetic_videos
from data_preprocessing import FAKE_BALANCE_TARGET, merge_synthetic_into_balanced_annotations, generate_synthetic_frame_annotations
import json

def render_synthetic_generation_ui():  

    st.markdown("## 🎭 Generate Synthetic Deepfake Videos")
    # Initialize session flags
    if "generation_started" not in st.session_state:
        st.session_state["generation_started"] = False
    if "stop_generation" not in st.session_state:
        st.session_state["stop_generation"] = False
    if "simswap_process" not in st.session_state:
        st.session_state["simswap_process"] = None
    
    # Generate Plan Button
    if st.button("🎬 Prepare Synthetic Generation Plan"):
        try:
            # Read data
            annotations_df = pd.read_csv("final_output/balanced_metadata.csv")
            utk_synthetic_df = pd.read_csv("final_output/synthetic_allocation.csv")
            
            # Prepare real and fake datasets
            real_df = utk_synthetic_df[(utk_synthetic_df["source"] == "UTKFace") & (utk_synthetic_df["label"] == "real")]
            fake_df = annotations_df[(annotations_df["label"] == "fake") & (annotations_df["source"].isin(["celeb", "faceforensics"]))]
    
            # Read balance config
            with open("final_output/balance_config.json", "r") as f:
                config = json.load(f)
                FAKE_BALANCE_TARGET = config.get("fake_balance_target")
    
            # Create synthetic generation plan
            synthetic_plan = {}
            for age_group in sorted(real_df["age_group"].unique()):
                current_fake = fake_df[fake_df["age_group"] == age_group].shape[0]
                required = max(FAKE_BALANCE_TARGET - current_fake, 0)
                if required > 0:
                    synthetic_plan[age_group] = required
    
            # If no synthetic videos are needed, display a success message
            if not synthetic_plan:
                st.success("✅ Fake videos are already balanced across all age groups.")
            else:
                # Store plan in session state
                st.session_state.synthetic_plan = synthetic_plan
                st.session_state.real_df_for_synthetic = real_df
    
                # Display synthetic generation plan
                st.markdown("#### 🧠 Synthetic Generation Plan")
                st.dataframe(pd.DataFrame({
                    "Age Group": list(synthetic_plan.keys()),
                    "Synthetic Videos to Create": list(synthetic_plan.values())
                }))
    
        except Exception as e:
            st.error(f"❌ Error preparing synthetic generation plan: {e}")
    
    
    # Button to trigger synthetic video generation - outside the plan display
    if st.button("🚀 Generate Synthetic Videos Now"):
        try:
            print("Button clicked!")  # This should print in the terminal or Streamlit's log
            
            # Get synthetic plan and real data for video generation
            synthetic_plan = st.session_state.get("synthetic_plan", {})
            real_df = st.session_state.get("real_df_for_synthetic", pd.DataFrame())
    
            if not synthetic_plan or real_df.empty:
                st.error("❌ No synthetic plan available or real data is missing.")
                return
    
            st.session_state["generation_started"] = True
            st.session_state["stop_generation"] = False
    
            selected_rows = []
            for age_group, num_videos in synthetic_plan.items():
                available = real_df[real_df["age_group"] == age_group]
                print(f"Age Group: {age_group} | Available Videos: {len(available)}") 
                if not available.empty:
                    selected = available.sample(n=min(num_videos, len(available)))
                    selected_rows.append(selected)
    
            print(f"Generating synthetic videos...")
    
            if selected_rows:
                final_df = pd.concat(selected_rows).reset_index(drop=True)
                print(f"Generating {len(final_df)} synthetic videos...")
                progress_bar = st.progress(0, text="Generating synthetic videos...")
    
                # Call the function to generate synthetic videos
                generate_synthetic_videos(final_df, streamlit_progress=progress_bar, st_module=st)
                progress_bar.empty()
                st.success("✅ Synthetic videos generated")
            else:
                st.warning("❌ No videos selected for generation.")
        
        except Exception as e:
            st.error(f"❌ Error generating synthetic videos: {e}")
            print(f"Error: {e}")

    #----------------------------------------------
    if st.button("🔄 Resume Incomplete Synthetic Generation"):
        st.session_state["generation_started"] = True
        st.session_state["stop_generation"] = False
    
        try:
            annotations_df = pd.read_csv("final_output/balanced_metadata.csv")
            real_df = annotations_df[(annotations_df["source"] == "UTKFace") & (annotations_df["label"] == "real")]
            if real_df.empty:
                st.warning("⚠️ No UTKFace real images available.")
                return
        except Exception as e:
            st.error(f"❌ Error loading metadata: {e}")
            return
    
        try:
            with open("final_output/balance_config.json", "r") as f:
                config = json.load(f)
                FAKE_BALANCE_TARGET = config.get("fake_balance_target")
        except Exception:
            st.error("❌ Could not load balance target. Run balancing first.")
            return
    
        resume_df = get_remaining_images_for_synthetic_generation(real_df, FAKE_BALANCE_TARGET)
        if resume_df.empty:
            st.success("✅ All synthetic videos already generated.")
        else:
            st.markdown("### 🔁 Synthetic Videos To Be Generated")
            st.markdown("#### 🧠 Synthetic Generation Plan")
        
            age_group_counts = resume_df["age_group"].value_counts().sort_index()
            st.dataframe(pd.DataFrame({
                "Age Group": age_group_counts.index,
                "Synthetic Videos to Create": age_group_counts.values
            }))
    
            progress_bar = st.progress(0, text="Resuming synthetic generation...")
            generate_synthetic_videos(resume_df, streamlit_progress=progress_bar, st_module=st)
            progress_bar.empty()
            st.success("✅ Resumed synthetic generation completed.")

    #----------------------------------------
    if os.path.exists("final_output/temp_synthetic_annotations.csv"):
        st.markdown("### 💾 Update Files with Syntehtic Data")
        if st.button("📥 Merge Saved Synthetic Metadata"):
            try:
                temp_df = pd.read_csv("final_output/temp_synthetic_annotations.csv")
                balance_metadata_path = "final_output/balanced_metadata.csv"
                metadata_path = "all_data_videos/annotations.csv"    
                existing_df = pd.read_csv(balance_metadata_path) if os.path.exists(balance_metadata_path) else pd.DataFrame()
                updated_df = pd.concat([existing_df, temp_df], ignore_index=True)
                updated_df.to_csv(balance_metadata_path, index=False)
                updated_df.to_csv(metadata_path, index=False)
                st.success(f"✅ Recovered and saved {len(temp_df)} synthetic entries.")

                frame_rate = st.session_state.get("selected_frame_rate", 10)
            
                merge_synthetic_into_balanced_annotations()
                generate_synthetic_frame_annotations(frame_rate = frame_rate)
                st.success("✅ All three metadata files created successfully ")
                # Clean up
                os.remove("final_output/temp_synthetic_annotations.csv")
            except Exception as e:
                st.error(f"❌ Recovery failed: {e}")
    
    