import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def get_age_label_source_table(annotation_file="all_data_videos/annotations.csv"):
    try:
        df = pd.read_csv(annotation_file)
        df["label_age"] = df["label"] + " (" + df["age_group"] + ")"

        pivot = pd.pivot_table(
            df,
            values="filename",
            index="label_age",
            columns="source",
            aggfunc="count",
            fill_value=0
        ).sort_index()

        return pivot
    except FileNotFoundError:
        return None

#------------------ Section for: VISUALIZING THE AGE DISTIBUTION --------------
def visualize_age_distribution(df):
    age_groups = df['age_group'].unique()
    num_groups = len(age_groups)

    # Dynamically scale width and height
    fig_width = max(6, num_groups * 2.5)
    fig_height = max(4, num_groups * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.countplot(x='age_group', data=df, order=['0-10','10-19','19-35','36-50','51+'], ax=ax)
    ax.set_title("Age Group Distribution")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")

    # Add count labels on top
    for p in ax.patches:
        height_val = int(p.get_height())
        ax.annotate(str(height_val),
                    (p.get_x() + p.get_width() / 2., height_val),
                    ha='center', va='bottom',
                    fontsize=10, xytext=(0, 5),
                    textcoords='offset points')

    plt.tight_layout()
    return fig

#------------------ Section for: VISUALIZING THE age distribution for SOURCE --------------

def show_age_distribution_pie_charts(annotation_file="all_data_videos/annotations.csv", return_figures=False):
    figures = []

    try:
        df = pd.read_csv(annotation_file)
        if "label" not in df.columns or "source" not in df.columns:
            st.error("Required columns `label` and `source` not found in annotations.")
            return []

        sources = df["source"].unique()
        label_colors = {"real": "#1f77b4", "fake": "#ff7f0e"}  # 🔵 blue, 🟠 orange

        if not return_figures:
            st.markdown("### 🥧 Age Group Distribution by Source and Label")
            st.markdown("🔵 **Real**  🟠 **Fake**")

        cols = st.columns(3)

        for i, source in enumerate(sorted(sources)):
            sub_df = df[df["source"] == source]
            label_counts = sub_df["label"].value_counts().reindex(["real", "fake"], fill_value=0)
            
            label_counts = label_counts[label_counts > 0]

            if label_counts.empty:
                continue  # Skip empty charts
            
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                label_counts.values,
                labels=label_counts.index,
                autopct='%1.1f%%',
                colors=[label_colors[label] for label in label_counts.index],
                startangle=90
            )
            ax.axis("equal")
            if return_figures:
                figures.append((source, fig))
            else:
                with cols[i % 3]:
                    st.pyplot(fig)

    except FileNotFoundError:
        if not return_figures:
            st.error("❌ `annotations.csv` not found. Please run age annotation first.")
        else:
            return []

    if return_figures:
        return figures
