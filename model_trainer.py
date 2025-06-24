import os
import cv2
import pandas as pd
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
# ------------------------------
# Dataset Loader for Image Frames
# ------------------------------
class FrameDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['path'])
    
        if image is None:
            raise FileNotFoundError(f"⚠️ Failed to load image: {row['path']}")
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        if self.transform:
            image = self.transform(image)
    
        label = 1 if row['label'] == 'fake' else 0
        return image, label



def copy_utkface_real_frame_location():  
    
    # --- Config ---
    annotations_csv = "final_output/frame_level_annotations.csv"
    source_folder = "all_data_videos/real_images"
    target_folder = "all_data_frames"
    # Load annotations
    df = pd.read_csv(annotations_csv)
    
    # Ensure target directory exists
    os.makedirs(target_folder, exist_ok=True)    
    def copy_utkface_real_image(row):
        
        filename = os.path.basename(row["frame"])
        source_path = os.path.join(source_folder, filename)
    
        if os.path.exists(source_path):
            new_path = os.path.join(target_folder, filename)
            shutil.copy2(source_path, new_path)
            return new_path
        else:
            return row["path"]  # Keep original path if file doesn't exist
    
    # Apply and update 'path' column
    df["path"] = df.apply(copy_utkface_real_image, axis=1)
    
    # Overwrite original CSV
    df.to_csv(annotations_csv, index=False)
    
    print(f"✅ Copied  files to `{target_folder}`.")
    print(f"📄 Updated annotations saved to: {annotations_csv}")
# ------------------------------
# Train/Test Split
# ------------------------------

def prepare_data_split(metadata_path):
    """
    Stratified split by age group and label, then:
    - Remove 'real'/'fake' from filenames
    - Rename physical files on disk to match new path
    - Drop 'source' column
    - Save train/test CSVs and rename mappings
    """
    

    os.makedirs("final_output", exist_ok=True)

    if os.path.exists("final_output/train_split.csv"):
        os.remove("final_output/train_split.csv")
    if os.path.exists("final_output/test_split.csv"):
        os.remove("final_output/test_split.csv")

    copy_utkface_real_frame_location()
    
    df = pd.read_csv(metadata_path)

    df["strata"] = df["age_group"].astype(str) + "_" + df["label"].astype(str)

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["strata"],
        random_state=42
    )

    train_df.drop(columns=["strata"], inplace=True)
    test_df.drop(columns=["strata"], inplace=True)

    def sanitize_and_rename(df, split_name):
        df = df.copy()
        df["original_path"] = df["path"]
        df["original_frame"] = df["frame"]

        def rename_path(old_path):
            new_path = old_path.replace("_real_", "_cls_").replace("_fake_", "_cls_")
            new_path = new_path.replace("real_", "cls_").replace("fake_", "cls_")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            else:
                print(f"❗ File not found: {old_path}")
            return new_path

        df["path"] = df["path"].apply(rename_path)
        df["frame"] = df["frame"].str.replace(r"_real_|_fake_", "_cls_", regex=True)
        df["frame"] = df["frame"].str.replace(r"real_|fake_", "cls_", regex=True)

        df.drop(columns=["source"], inplace=True)

        # Save updated split and mapping
        df.to_csv(f"final_output/{split_name}_split.csv", index=False)
        df[["original_frame", "frame", "label"]].to_csv(f"final_output/{split_name}_frame_rename_mapping.csv", index=False)
        return df

    train_df = sanitize_and_rename(train_df, "train")
    test_df = sanitize_and_rename(test_df, "test")

    return train_df, test_df

# ------------------------------
# Model Factory
# ------------------------------
def get_model(model_name):
    if model_name == "XceptionNet":
        model = timm.create_model("xception", pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "EfficientNet":
        model = timm.create_model("efficientnet_b0", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == "LipForensics":
        model = timm.create_model("mobilenetv2_100", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError("Unsupported model")
    return model

# ------------------------------
# Model Training 
# ------------------------------

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

def train_model(model_name, train_df, transform, streamlit_mode=False, suffix=None):
    import time
    if streamlit_mode:
        import streamlit as st
        st.markdown(f"### 🏋️ Training: `{model_name}`")

    model = get_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Split into train/val
    train_split, val_split = train_test_split(train_df, test_size=0.3, stratify=train_df["label"], random_state=42)
    train_dataset = FrameDataset(train_split, transform)
    val_dataset = FrameDataset(val_split, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()

    num_epochs = 20
    total_steps = num_epochs * len(train_loader)
    current_step = 0

    if streamlit_mode:
        progress_bar = st.progress(0)
        status_text = st.empty()

    best_val_auc = 0.0
    patience = 3
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
    
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
    
            optimizer.zero_grad()
            with autocast():
                outputs = model(images).squeeze(dim=1)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            total_loss += loss.item()
            current_step += 1
    
            if streamlit_mode:
                progress_bar.progress(current_step / total_steps)
    
        avg_train_loss = total_loss / len(train_loader)
    
        # 🔍 Validation
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(images).squeeze(dim=1)
                    probs = torch.sigmoid(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
    
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except:
            val_auc = 0.0
    
        # 🧠 Early stopping logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            suffix_str = f"_{suffix}" if suffix else ""
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best{suffix_str}.pth")  # Save best
        else:
            epochs_no_improve += 1
    
        log_msg = f"📊 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}"
        print(log_msg)
        if streamlit_mode:
            status_text.markdown(log_msg)
        time.sleep(0.1)
    
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}. Best AUC: {best_val_auc:.4f}")
            if streamlit_mode:
                st.warning(f"🛑 Early stopping at epoch {epoch+1}. Best AUC: {best_val_auc:.4f}")
            early_stop = True
            break

    if streamlit_mode:
        st.success(f"✅ Finished training `{model_name}`")
        progress_bar.empty()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{model_name}_best{suffix_str}.pth")

    return model

# ------------------------------
# Model Training Wrapper
# ------------------------------
def train_models(selected_models, train_csv, streamlit_mode=False):
    train_df = pd.read_csv(train_csv)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    trained_models = {}

    if streamlit_mode:
        import streamlit as st
        st.markdown("### 🏋️‍♂️ Training Models")

    for model_name in selected_models:
        model = train_model(model_name, train_df, transform, streamlit_mode=streamlit_mode)
        trained_models[model_name] = model

    return trained_models

# ------------------------------
# Model Training For celeb and FaceForensics
# ------------------------------

def train_models_on_source(source_name, metadata_csv, selected_models, streamlit_mode=False):
    import time
    from torch.cuda.amp import autocast, GradScaler

    df = pd.read_csv(metadata_csv)
    source_df = df[df["source"] == source_name]

    train_df, val_df = train_test_split(
        source_df, test_size=0.3, stratify=source_df["label"], random_state=42
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if streamlit_mode:
        import streamlit as st
        st.markdown(f"### 🧪 Training on Source: `{source_name}`")

    for model_name in selected_models:
        print(f"🔄 Training {model_name} on {source_name}...")
        if streamlit_mode:
            st.write(f"🔄 Training **{model_name}**...")

        model = get_model(model_name).to(device)

        train_loader = DataLoader(FrameDataset(train_df, transform), batch_size=32, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(FrameDataset(val_df, transform), batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scaler = GradScaler()

        num_epochs = 20
        best_val_auc = 0.0
        epochs_no_improve = 0
        patience = 3

        if streamlit_mode:
            progress_bar = st.progress(0)
            status_text = st.empty()

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(images).squeeze(dim=1)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            all_labels, all_preds = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with autocast():
                        outputs = model(images).squeeze(dim=1)
                        probs = torch.sigmoid(outputs)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(probs.cpu().numpy())

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.0

            # Early stopping logic
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/{model_name}_{source_name}_best.pth")
            else:
                epochs_no_improve += 1

            log_msg = f"📊 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}"
            print(log_msg)
            if streamlit_mode:
                status_text.markdown(log_msg)
            time.sleep(0.1)

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}. Best AUC: {best_val_auc:.4f}")
                if streamlit_mode:
                    st.warning(f"🛑 Early stopping at epoch {epoch+1}. Best AUC: {best_val_auc:.4f}")
                break

        # Save final model
        torch.save(model.state_dict(), f"checkpoints/{model_name}_{source_name}_final.pth")
        results[model_name] = model

        if streamlit_mode:
            st.success(f"✅ Finished training `{model_name}` on `{source_name}`")
            progress_bar.empty()

    return results, val_df






