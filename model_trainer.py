import os
import cv2
import pandas as pd
import numpy as np
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

# ------------------------------
# Train/Test Split
# ------------------------------

def prepare_data_split(metadata_path):
    """
    Stratified split by both age group and label.
    """
    # 🧹 Remove old splits if they exist
    if os.path.exists("train_split.csv"):
        os.remove("train_split.csv")
    if os.path.exists("test_split.csv"):
        os.remove("test_split.csv")

    df = pd.read_csv(metadata_path)

    # Create stratification key: "agegroup_label" (e.g., "20–30_1")
    df["strata"] = df["age_group"].astype(str) + "_" + df["label"].astype(str)

    # Stratified split on combined strata
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["strata"],
        random_state=42
    )

    # Drop helper column
    train_df.drop(columns=["strata"], inplace=True)
    test_df.drop(columns=["strata"], inplace=True)

    # Save results
    train_df.to_csv("train_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False)


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
def train_model(model_name, train_df, transform, streamlit_mode=False):
    print(f"\n🚀 Training: {model_name}")
    model = get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_dataset = FrameDataset(train_df, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    total_steps = 3 * len(train_loader)
    current_step = 0

    if streamlit_mode:
        import streamlit as st
        progress_bar = st.progress(0)
        st.write(f"📚 Training: **{model_name}**")

    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_step += 1
            if streamlit_mode:
                progress_bar.progress(current_step / total_steps)

    if streamlit_mode:
        progress_bar.empty()

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
    df = pd.read_csv(metadata_csv)
    source_df = df[df["source"] == source_name]

    # Split the source data (70% train, 30% test)
    train_df, test_df = train_test_split(
        source_df, test_size=0.3, stratify=source_df["label"], random_state=42
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if streamlit_mode:
        import streamlit as st
        st.markdown(f"### 🧪 Training on Source")

    for model_name in selected_models:
        if streamlit_mode:
            st.write(f"🔄 Training **{model_name}**...")
        print(f"🔄 Training {model_name}...")

        model = get_model(model_name).to(device)
        model.train()

        dataset = FrameDataset(train_df, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(3):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        results[model_name] = model

    return results, test_df





