import os
import json
import pandas as pd
from synthetic_data_simswap import generate_synthetic_videos

def resume_synthetic_generation():
    metadata_path = "all_data_videos/annotations.csv"
    balanced_path = "final_output/balanced_metadata.csv"
    config_path = "final_output/balance_config.json"
    real_image_dir = "all_data_videos/real_images"
    synthetic_dir = "all_data_videos/synthetic"

    if not os.path.exists(balanced_path):
        print("❌ Balanced metadata file not found.")
        return

    if not os.path.exists(config_path):
        print("❌ balance_config.json not found. Run the balancing step first.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
        FAKE_BALANCE_TARGET = config.get("fake_balance_target")

    if FAKE_BALANCE_TARGET is None:
        print("❌ FAKE_BALANCE_TARGET not set in config.")
        return

    df = pd.read_csv(balanced_path)
    real_df = df[(df["label"] == "real") & (df["source"] == "UTKFace")]
    fake_df = df[(df["label"] == "fake") & (df["source"].isin(["celeb", "faceforensics", "synthetic"]))]

    # Extract already generated synthetic filenames
    already_generated = set(os.path.splitext(f)[0] for f in os.listdir(synthetic_dir) if f.endswith((".mp4", ".avi")))

    plan = {}
    for age_group in sorted(real_df["age_group"].unique()):
        current_fake = fake_df[fake_df["age_group"] == age_group].shape[0]
        needed = max(FAKE_BALANCE_TARGET - current_fake, 0)

        if needed > 0:
            plan[age_group] = needed

    if not plan:
        print("✅ All age groups already have enough fake videos.")
        return

    # Refilter real_df to exclude ones already used (based on filename logic)
    selected_rows = []
    for age_group, needed in plan.items():
        available = real_df[real_df["age_group"] == age_group]
        remaining = []

        for row in available.itertuples(index=False):
            image_base = row.filename.replace("utkface_real_", "").replace(".jpg", "").replace(".png", "")
            expected_prefix = f"utkface_fake_{image_base}_on_"
            if not any(fname.startswith(expected_prefix) for fname in already_generated):
                remaining.append(row)

        if not remaining:
            print(f"⚠️ Not enough unused images for age group {age_group}")
            continue

        sampled = pd.DataFrame(remaining).sample(n=min(needed, len(remaining)), random_state=42)
        selected_rows.append(sampled)

    if not selected_rows:
        print("⚠️ No remaining images available for resuming generation.")
        return

    to_generate_df = pd.concat(selected_rows).reset_index(drop=True)
    print(f"🔁 Resuming generation for {len(to_generate_df)} synthetic videos.")
    generate_synthetic_videos(to_generate_df)

if __name__ == "__main__":
    resume_synthetic_generation()
