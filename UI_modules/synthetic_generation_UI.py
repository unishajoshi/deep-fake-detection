import streamlit as st
import os
import pandas as pd
from logger import log_action
from synthetic_data_simswap import generate_synthetic_videos

import streamlit as st
import os
import pandas as pd
from logger import log_action
from synthetic_data_simswap import generate_synthetic_videos

def render_synthetic_generation_ui():
    st.markdown("## 🎭 Generate Synthetic Deepfake Videos")

    try:
        annotations_df = pd.read_csv("balanced_annotations.csv")
    except FileNotFoundError:
        st.error("❌ No metadata file found (balanced_annotations.csv). Cannot proceed.")
        return

    # Filter UTKFace real images for synthesis
    real_df = annotations_df[annotations_df["source"] == "UTKFace"]
    if real_df.empty:
        st.warning("⚠️ No real images found with source 'UTKFace'.")
        return

    # Filter fake videos from Celeb and FaceForensics++
    celeb_fake_df = annotations_df[
        (annotations_df["label"] == "fake") &
        (annotations_df["source"].isin(["celeb", "faceforensics++"]))
    ]

    # Count fake samples by age group and find the maximum available
    fake_counts = celeb_fake_df["age_group"].value_counts()
    max_fake = fake_counts.max()

    synthetic_plan = {}
    for age_group in sorted(real_df["age_group"].unique()):
        current_fake = fake_counts.get(age_group, 0)
        required = max(max_fake - current_fake, 0)
        if required > 0:
            synthetic_plan[age_group] = required

    if not synthetic_plan:
        st.success("✅ Fake videos are already balanced across all age groups.")
        return

    st.markdown("#### 📝 Select Synthetic Videos to Generate")
    user_inputs = {}

    for age_group, default_num in synthetic_plan.items():
        available_images = real_df[real_df["age_group"] == age_group]
        max_available = len(available_images)
        max_selectable = min(default_num * 2, max_available)

        user_inputs[age_group] = st.selectbox(
        f"Age Group {age_group} - Synthetic Videos to Create",
        options=list(range(0, int(max_selectable) + 1)),
        index=int(default_num) if default_num <= max_selectable else int(max_selectable)
        )

    # 🔍 Show preview summary table
    if user_inputs:
        preview_data = {
            "Age Group": list(user_inputs.keys()),
            "Existing Fake Videos": [fake_counts.get(ag, 0) for ag in user_inputs.keys()],
            "Synthetic Videos to Create": list(user_inputs.values())
        }
        preview_df = pd.DataFrame(preview_data)
        st.markdown("#### Existing Videos Vs Synthetic Video to Create")
        st.dataframe(preview_df)

    # 🔘 Trigger generation
    if st.button("🎬 Create Synthetic Videos"):
        selected_rows = []
        for age_group, num_videos in user_inputs.items():
            group_df = real_df[real_df["age_group"] == age_group]
            if not group_df.empty and num_videos > 0:
                samples = group_df.sample(n=min(len(group_df), num_videos))
                selected_rows.append(samples)

        if selected_rows:
            final_df = pd.concat(selected_rows).reset_index(drop=True)
            progress_bar = st.progress(0, text="Generating synthetic videos...")
            generate_synthetic_videos(final_df, streamlit_progress=progress_bar, st_module=st)
            progress_bar.empty()
            st.success("✅ All synthetic videos generated successfully!")
        else:
            st.warning("⚠️ No samples selected for generation.")
