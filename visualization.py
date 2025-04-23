import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def get_age_label_source_table(annotation_file="annotations.csv"):
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
def show_age_distribution_pie_charts(annotation_file="annotations.csv", return_figures=False):
    figures = []

    try:
        df = pd.read_csv(annotation_file)

        if not return_figures:
            st.markdown("### 🥧 Age Group Distribution by Source")
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

        # celeb chart
        celeb_df = df[df["source"] == "celeb"]
        if not celeb_df.empty:
            fig1, ax1 = plt.subplots()
            celeb_counts = celeb_df['age_group'].value_counts().sort_index()
            ax1.pie(celeb_counts, labels=celeb_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')

            if return_figures:
                figures.append(("celeb", fig1))
            else:
                with col2:
                    st.markdown("#### 🟢 celeb")
                    st.pyplot(fig1)

        # FaceForensics++ chart
        ff_df = df[df["source"] == "faceforensics"]
        if not ff_df.empty:
            fig2, ax2 = plt.subplots()
            ff_counts = ff_df['age_group'].value_counts().sort_index()
            ax2.pie(ff_counts, labels=ff_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')

            if return_figures:
                figures.append(("FaceForensics++", fig2))
            else:
                with col3:
                    st.markdown("#### 🔵 FaceForensics++")
                    st.pyplot(fig2)

    except FileNotFoundError:
        if not return_figures:
            st.error("❌ `annotations.csv` not found. Please run age annotation first.")
        else:
            return []

    if return_figures:
        return figures