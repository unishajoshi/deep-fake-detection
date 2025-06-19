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
        required_cols = {"label", "source", "age"}
        if not required_cols.issubset(df.columns):
            st.error(f"Required columns {required_cols} not found in annotations.")
            return []

        # Define age bins and labels
        bins = [0, 18, 30, 45, 60, 100]
        labels = ["0-10", "10-18", "19-35", "35-50", "51+"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

        if not return_figures:
            st.markdown("### 🥧 Age Group Distribution by Source")

        sources = df["source"].unique()
        cols = st.columns(3)

        for i, source in enumerate(sorted(sources)):
            sub_df = df[df["source"] == source]
            age_group_counts = sub_df["age_group"].value_counts().sort_index()

            # Remove age groups with zero count
            age_group_counts = age_group_counts[age_group_counts > 0]

            if age_group_counts.empty:
                continue

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                age_group_counts,
                labels=age_group_counts.index,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                startangle=90
            )
            #ax.set_title(f"{source}")
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
