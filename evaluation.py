import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader
from model_trainer import FrameDataset, get_model
from torch.cuda.amp import autocast  


# ------------------------------
# ERR calculation
# ------------------------------

def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    diff = fpr - fnr
    idx = (diff ** 2).argmin()
    return round((fpr[idx] + fnr[idx]) / 2, 4)

# ------------------------------
# Model Evaluation
# ------------------------------

def evaluate_model(model, test_df_dict, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    results = {}

    for label, df_subset in test_df_dict.items():
        dataset = FrameDataset(df_subset, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        all_labels, all_scores = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(images).squeeze(dim=1)
                    scores = torch.sigmoid(outputs)
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_scores.extend(scores.view(-1).cpu().numpy())

        if len(set(all_labels)) < 2:
            results[label] = {"auc": None, "pauc": None, "eer": None}
        else:
            auc = round(roc_auc_score(all_labels, all_scores), 4)
            pauc = round(average_precision_score(all_labels, all_scores), 4)
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            eer = calculate_eer(fpr, tpr)
            results[label] = {"auc": auc, "pauc": pauc, "eer": eer}

    return results
# ------------------------------
# Results for Grouped by Dataset
# ------------------------------
def flatten_results_grouped(results_by_dataset):
    summary = {}
    for model_name, datasets in results_by_dataset.items():
        for dataset_name, metrics in datasets.items():
            if dataset_name not in summary:
                summary[dataset_name] = {}
            for metric in ["auc", "pauc", "eer"]:
                summary[dataset_name][(model_name, metric.upper())] = metrics.get(metric)

    df = pd.DataFrame.from_dict(summary, orient="index")
    df.index.name = "Test Set"
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Model", "Metric"])
    return df.reset_index()

# ------------------------------
# Results for Grouped by Age group
# ------------------------------
def flatten_age_specific_results(results_by_model):
    rows, index = [], []

    for model_name, datasets in results_by_model.items():
        dataset_names = set(k[0] for k in datasets.keys())
        for dataset_name in dataset_names:
            row, index_entry = [], (model_name, dataset_name)
            index.append(index_entry)

            for age_group in ["overall", "0-10", "10-18", "19-35", "36-50","51+"]:
                metrics = datasets.get((dataset_name, age_group), {})
                row.extend([metrics.get("auc"), metrics.get("pauc"), metrics.get("eer")])
            rows.append(row)

    columns = pd.MultiIndex.from_product(
    [["overall", "0-10", "10-18", "19-35", "36-50","51+"], ["AUC", "PAUC", "EER"]],
        names=["Age Group", "Metric"]
    )
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["Model", "Dataset"]), columns=columns)

# ------------------------------
# Evaluation for all Datasets
# ------------------------------

def evaluate_on_all_sets(selected_models, streamlit_mode=False):
    import shutil

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and sanitize full dataset
    full_original = pd.read_csv("final_output/frame_level_annotations_source.csv")

    if "original_path" not in full_original.columns:
        full_original["original_path"] = full_original["path"]
        full_original["original_frame"] = full_original["frame"]

        def sanitize_row(row):
            old_path = row["original_path"]
            new_path = old_path.replace("_real_", "_cls_").replace("_fake_", "_cls_")
            new_path = new_path.replace("real_", "cls_").replace("fake_", "cls_")

            if os.path.exists(old_path) and old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"⚠️ Rename failed: {old_path} → {new_path} :: {e}")
            return new_path

        full_original["path"] = full_original.apply(sanitize_row, axis=1)
        full_original["frame"] = full_original["frame"].str.replace(r"_real_|_fake_", "_cls_", regex=True)
        full_original["frame"] = full_original["frame"].str.replace(r"real_|fake_", "cls_", regex=True)

        full_original.to_csv("final_output/frame_level_annotations_source.csv", index=False)
        full_original[["original_frame", "frame", "label", "source"]].to_csv(
            "final_output/frame_level_annotations_mapping.csv", index=False
        )
        print("✅ Frame-level annotations sanitized.")

    test_balanced = pd.read_csv("final_output/test_split.csv")
    n_test = len(test_balanced)

    celeb_all = full_original[full_original["source"] == "celeb"]
    ffpp_all = full_original[full_original["source"] == "faceforensics"]

    celeb_test = celeb_all.sample(n=min(n_test, len(celeb_all)), random_state=42)
    ffpp_test = ffpp_all.sample(n=min(n_test, len(ffpp_all)), random_state=42)

    subsets = {
        "Balanced": test_balanced,
        "Celeb": celeb_test,
        "FaceForensics++": ffpp_test
    }

    results = {}

    for model_name in selected_models:
        if streamlit_mode:
            import streamlit as st
            st.markdown(f"#### 🔍 Evaluating {model_name}")
        print(f"🔍 Evaluating {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(model_name)
        model.load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth", map_location=device))
        model = model.to(device)
        results[model_name] = evaluate_model(model, subsets, transform)

    return results

# ------------------------------
# Evaluation for all Datasets by age group
# ------------------------------

def evaluate_on_balanced_set_agewise(selected_models, transform=None, streamlit_mode=False):
    import pandas as pd
    import torch
    from torchvision import transforms

    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Load test data
    balanced_df = pd.read_csv("final_output/test_split.csv")
    original_df = pd.read_csv("final_output/frame_level_annotations_source.csv")
    n_test = len(balanced_df)

    age_groups = ["0-10", "10-18", "19-35", "36-50", "51+"]

    def get_age_splits(df):
        return {
            group: df[df["age_group"] == group]
            for group in age_groups
            if not df[df["age_group"] == group].empty
        }

    def safe_sample(df, n, seed):
        if len(df) == 0:
            return pd.DataFrame(columns=df.columns)
        return df.sample(n=min(n, len(df)), random_state=seed)

    celeb_df = original_df[original_df["source"] == "celeb"]
    ffpp_df = original_df[original_df["source"] == "faceforensics"]

    datasets = {
        "balance": balanced_df,
        "celeb": safe_sample(celeb_df, n_test, seed=1),
        "faceforensics": safe_sample(ffpp_df, n_test, seed=2),
    }

    all_results = {}

    for model_name in selected_models:
        model = get_model(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth", map_location=device))
        model = model.to(device)
        model.eval()

        model_results = {}

        for dataset_name, df in datasets.items():
            # ✅ Evaluate overall with tuple key
            model_results[(dataset_name, "overall")] = evaluate_model(model, {"overall": df}, transform).get("overall", {
                "auc": None, "pauc": None, "eer": None
            })

            # ✅ Evaluate each age group with tuple keys
            age_split = get_age_splits(df)
            for group, group_df in age_split.items():
                group_metrics = evaluate_model(model, {group: group_df}, transform)
                model_results[(dataset_name, group)] = group_metrics.get(group, {
                    "auc": None, "pauc": None, "eer": None
                })

        all_results[model_name] = model_results

    return all_results

# ------------------------------
# Evaluation for all Trained with original datasets
# ------------------------------
def evaluate_on_all_sets_for_trained_models(selected_models, source_name, streamlit_mode=False):
    import shutil

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and sanitize full dataset
    full_original = pd.read_csv("final_output/frame_level_annotations_source.csv")

    if "original_path" not in full_original.columns:
        full_original["original_path"] = full_original["path"]
        full_original["original_frame"] = full_original["frame"]

        def sanitize_row(row):
            old_path = row["original_path"]
            new_path = old_path.replace("_real_", "_cls_").replace("_fake_", "_cls_")
            new_path = new_path.replace("real_", "cls_").replace("fake_", "cls_")

            if os.path.exists(old_path) and old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"⚠️ Rename failed: {old_path} → {new_path} :: {e}")
            return new_path

        full_original["path"] = full_original.apply(sanitize_row, axis=1)
        full_original["frame"] = full_original["frame"].str.replace(r"_real_|_fake_", "_cls_", regex=True)
        full_original["frame"] = full_original["frame"].str.replace(r"real_|fake_", "cls_", regex=True)

        full_original.to_csv("final_output/frame_level_annotations_source.csv", index=False)
        full_original[["original_frame", "frame", "label", "source"]].to_csv(
            "final_output/frame_level_annotations_mapping.csv", index=False
        )
        print("✅ Frame-level annotations sanitized.")

    test_balanced = pd.read_csv("final_output/test_split.csv")
    n_test = len(test_balanced)

    celeb_all = full_original[full_original["source"] == "celeb"]
    ffpp_all = full_original[full_original["source"] == "faceforensics"]

    celeb_test = celeb_all.sample(n=min(n_test, len(celeb_all)), random_state=42)
    ffpp_test = ffpp_all.sample(n=min(n_test, len(ffpp_all)), random_state=42)

    subsets = {
        "Balanced": test_balanced,
        "Celeb": celeb_test,
        "FaceForensics++": ffpp_test
    }

    results = {}

    for model_name in selected_models:
        if streamlit_mode:
            import streamlit as st
            st.markdown(f"#### 🔍 Evaluating {model_name} on {source_name}")
        print(f"🔍 Evaluating {model_name} on {source_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load source-trained model
        model = get_model(model_name).to(device)
        checkpoint_path = f"checkpoints/{model_name}_{source_name}_best.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        model_results = evaluate_model(model, subsets, transform)
        results[f"{model_name} ({source_name})"] = model_results

    return results
