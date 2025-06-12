import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader
from model_trainer import FrameDataset, get_model


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
    model.eval()
    results = {}

    for label, df_subset in test_df_dict.items():
        dataset = FrameDataset(df_subset, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_labels, all_scores = [], []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
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
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()
    ])

    test_balanced = pd.read_csv("final_output/test_split.csv")
    full_original = pd.read_csv("final_output/frame_level_annotations_source.csv")
    n_test = len(test_balanced)

    celeb_all = full_original[full_original["source"] == "celeb"]
    ffpp_all = full_original[full_original["source"] == "faceforensics"]

    celeb_test = celeb_all.sample(n=min(n_test, len(celeb_all)), random_state=42)
    ffpp_test = ffpp_all.sample(n=min(n_test, len(ffpp_all)), random_state=42)

    subsets = {
        "Balanced": test_balanced,
        "Colab": celeb_test,
        "FaceForensics++": ffpp_test
    }

    results = {}
    for model_name in selected_models:
        if streamlit_mode:
            import streamlit as st
            st.markdown(f"#### 🔍 Evaluating {model_name}")
        print(f"🔍 Evaluating {model_name}")
        model = get_model(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        results[model_name] = evaluate_model(model, subsets, transform)

    return results

# ------------------------------
# Evaluation for all Datasets by age group
# ------------------------------

def evaluate_on_all_sets_agewise(selected_models, transform=None, streamlit_mode=False):
    import pandas as pd
    from torch.utils.data import DataLoader

    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    balanced_df = pd.read_csv("final_output/test_split.csv")
    original_df = pd.read_csv("final_output/frame_level_annotations_source.csv")
    n_test = len(balanced_df)

    age_groups = ["0-10", "10-19", "19-35", "36-50", "51+"]

    def get_age_splits(source_df):
        groups = {}
        for group in age_groups:
            subset = source_df[source_df["age_group"] == group]
            if len(subset) > 0:
                groups[group] = subset
        return groups

    def safe_sample(df, n, seed):
        if len(df) == 0:
            return pd.DataFrame(columns=df.columns)
        return df.sample(n=min(n, len(df)), random_state=seed)

    celeb_data = original_df[original_df["source"] == "celeb"]
    ffpp_data = original_df[original_df["source"] == "faceforensics"]

    datasets = {
        "balance": get_age_splits(balanced_df),
        "celeb": get_age_splits(safe_sample(celeb_data, n_test, seed=1)),
        "faceforensics": get_age_splits(safe_sample(ffpp_data, n_test, seed=2)),
    }

    all_results = {}

    for model_name in selected_models:
        model = get_model(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        model_results = {}

        for dataset_name, age_group_map in datasets.items():
            # ✅ Compute overall results
            if age_group_map:
                full_df = pd.concat(age_group_map.values(), ignore_index=True)
                loader = DataLoader(FrameDataset(full_df, transform), batch_size=32, shuffle=False)

                all_labels, all_scores = [], []
                with torch.no_grad():
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images).squeeze()
                        scores = torch.sigmoid(outputs)
                        all_labels.extend(labels.view(-1).cpu().numpy())
                        all_scores.extend(scores.view(-1).cpu().numpy())

                if len(set(all_labels)) < 2:
                    model_results[(dataset_name, "overall")] = {"auc": None, "pauc": None, "eer": None}
                else:
                    auc = round(roc_auc_score(all_labels, all_scores), 4)
                    pauc = round(average_precision_score(all_labels, all_scores), 4)
                    fpr, tpr, _ = roc_curve(all_labels, all_scores)
                    eer = calculate_eer(fpr, tpr)
                    model_results[(dataset_name, "overall")] = {"auc": auc, "pauc": pauc, "eer": eer}

            # ✅ Evaluate age-group subsets
            for group, df_subset in age_group_map.items():
                if df_subset.empty:
                    model_results[(dataset_name, group)] = {"auc": None, "pauc": None, "eer": None}
                    continue

                loader = DataLoader(FrameDataset(df_subset, transform), batch_size=32, shuffle=False)
                all_labels, all_scores = [], []
                with torch.no_grad():
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images).squeeze()
                        scores = torch.sigmoid(outputs)
                        all_labels.extend(labels.view(-1).cpu().numpy())
                        all_scores.extend(scores.view(-1).cpu().numpy())

                if len(set(all_labels)) < 2:
                    model_results[(dataset_name, group)] = {"auc": None, "pauc": None, "eer": None}
                else:
                    auc = round(roc_auc_score(all_labels, all_scores), 4)
                    pauc = round(average_precision_score(all_labels, all_scores), 4)
                    fpr, tpr, _ = roc_curve(all_labels, all_scores)
                    eer = calculate_eer(fpr, tpr)
                    model_results[(dataset_name, group)] = {"auc": auc, "pauc": pauc, "eer": eer}

        all_results[model_name] = model_results

    return all_results

# ------------------------------
# Evaluation for all Trained with original datasets
# ------------------------------
def evaluate_on_all_sets_for_trained_models(trained_models, streamlit_mode=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_balanced = pd.read_csv("final_output/test_split.csv")
    full_original = pd.read_csv("final_output/frame_level_annotations_source.csv")
    n_test = len(test_balanced)

    celeb_all = full_original[full_original["source"] == "celeb"]
    celeb_test = celeb_all.sample(n=min(n_test, len(celeb_all)), random_state=42)

    ffpp_all = full_original[full_original["source"] == "faceforensics"]
    ffpp_test = ffpp_all.sample(n=min(n_test, len(ffpp_all)), random_state=42)

    subsets = {
        "Balanced": test_balanced,
        "celeb": celeb_test,
        "FaceForensics++": ffpp_test
    }

    results = {}

    for model_name, model in trained_models.items():
        if streamlit_mode:
            import streamlit as st
            st.markdown(f"#### 🔍 Evaluating {model_name}")
        print(f"🔍 Evaluating {model_name}")

        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model_results = evaluate_model(model, subsets, transform)
        results[model_name] = model_results

    return results