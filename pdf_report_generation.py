from fpdf import FPDF
import pandas as pd
import os
import cv2
import streamlit as st
import tempfile
from PIL import Image

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", "fonts/DejaVuLGCSans.ttf", uni=True)
        self.add_font("DejaVu", "B", "fonts/DejaVuLGCSans-Bold.ttf", uni=True)
        self.set_font("DejaVu", "", 11)
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Deepfake Dataset Pipeline Report", ln=True, align="C")
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("DejaVu", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 11)

    def add_table(self, df):
        self.set_font("DejaVu", "", 6)
        epw = self.w - 2 * self.l_margin 
        col_width = epw / len(df.columns)
        self.set_fill_color(220, 220, 220)

        # Header
        for col in df.columns:
            self.cell(col_width, 8, str(col), border=1, fill=True)
        self.ln()

        # Rows
        for _, row in df.iterrows():
            for val in row:
                self.cell(col_width, 8, str(val), border=1)
            self.ln()

    def add_image_grid(self, image_paths, title, grid=(2, 3), img_size=60):
        self.section_title(title)
        x_start = self.get_x()
        y = self.get_y()

        count = 0
        for i, path in enumerate(image_paths):
            if count and count % grid[1] == 0:
                y += img_size + 10
                self.set_y(y)
                self.set_x(x_start)

            if os.path.exists(path):
                self.image(path, w=img_size)
            else:
                self.cell(img_size, img_size, "Image not found", border=1, align="C")

            self.set_x(self.get_x() + img_size + 5)
            count += 1
        self.ln(img_size + 10)


def safe_str(text):
    return str(text).encode('latin-1', 'ignore').decode('latin-1')

def draw_table_paginated(pdf, df, rows_per_page=25, cols_per_page=6):
    pdf.set_font("DejaVu", "", 6)
    
    table = df.astype(str).values.tolist()
    headers = list(df.columns)
    table.insert(0, headers)

    print_w = pdf.w - 2 * pdf.l_margin
    print_h = pdf.h - 2 * pdf.t_margin - 20
    c_h = print_h / rows_per_page
    c_w = print_w / cols_per_page

    num_rows = len(table)
    num_cols = len(table[0])

    row_offset = 0
    while row_offset < num_rows:
        row_max = min(row_offset + rows_per_page, num_rows)
        col_offset = 0

        while col_offset < num_cols:
            col_max = min(col_offset + cols_per_page, num_cols)
            pdf.set_font("DejaVu", "", 6)

            for i in range(row_offset, row_max):
                for j in range(col_offset, col_max):
                    val = table[i][j]
                    #pdf.cell(c_w, c_h, val[:20], border=1, ln=0, align="C")
                    pdf.cell(c_w, c_h, safe_str(val[:20]), border=1, ln=0, align="C")
                pdf.ln()

            col_offset += cols_per_page
        row_offset += rows_per_page

def create_full_pdf_report(
    output_path="final_output/final_report.pdf",
    dataset_summary=None,
    cleaning_notes=None,
    frame_count_summary=None,
    real_frames=[],
    fake_frames=[],
    sample_frame_paths=[],
    age_annotation_summary=None,
    balance_summary=None,
    gradcam_paths=[]
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Dataset Summary
    pdf.section_title("Dataset Summary")
    if dataset_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(dataset_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Dataset summary not available.", ln=True)

    # 2. Cleaning Summary
    pdf.section_title("Cleaning Summary")
    pdf.multi_cell(0, 10, cleaning_notes or "No cleaning performed.")

    # 3. Frame Extraction
    pdf.section_title("Frame Extraction")
    #pdf.multi_cell(0, 10, str(frame_count_summary) if frame_count_summary is not None else "No frame extraction summary available.")
    pdf.multi_cell(0, 10, f"{frame_count_summary} frames were extracted in total from all available videos." if frame_count_summary is not None else "No frame extraction summary available.")

    # 4. Sample Frames
    pdf.section_title("Frame Extraction")
    if real_frames:
        # Title
        pdf.cell(0, 10, "Real Frames Samples", ln=True)

        # Layout parameters
        image_width = 60  # mm
        image_height = 45  # mm
        spacing = 5  # mm
        margin_x = pdf.l_margin
        start_y = pdf.get_y()
        x = margin_x
        y = start_y

        for i, img_path in enumerate(real_frames):
            pdf.image(img_path, x=x, y=y, w=image_width, h=image_height)

            # Move to the next column
            x += image_width + spacing

            # If 3 images in a row, break
            if (i + 1) % 3 == 0:
                break  # stop at 3 images
        
        pdf.set_y(y + image_height + spacing)
    else:
        pdf.cell(0, 10, "No real frames available.", ln=True)
        
    if fake_frames:
        # Title
        pdf.cell(0, 10, "Fake Frames Samples", ln=True)

        # Layout parameters
        image_width = 60  # mm
        image_height = 45  # mm
        spacing = 5  # mm
        margin_x = pdf.l_margin
        start_y = pdf.get_y()
        x = margin_x
        y = start_y

        for i, img_path in enumerate(fake_frames):
            pdf.image(img_path, x=x, y=y, w=image_width, h=image_height)

            # Move to the next column
            x += image_width + spacing

            # If 3 images in a row, break
            if (i + 1) % 3 == 0:
                break  # stop at 3 images
        
        pdf.set_y(y + image_height + spacing)
    else:
        pdf.cell(0, 10, "No fake frames available.", ln=True)
        
    # 5. Age Annotation Summary
    pdf.section_title("Age Annotation Summary")
    if age_annotation_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(age_annotation_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Age annotation not available.", ln=True)
    
    # Save and add pie charts
    if "pie_chart_figures" in st.session_state:
        for source, fig in st.session_state.pie_chart_figures:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{source.capitalize()} Dataset", ln=True)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                tmp_path = tmpfile.name

            fig.savefig(tmp_path, bbox_inches='tight')

            pdf.image(tmp_path, w=100)

            os.remove(tmp_path)
    else:
        pdf.multi_cell(0, 10, "No pie chart figures available.")
    # 6. Balance Summary
    pdf.section_title("Balanced Dataset Summary")
    if balance_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(balance_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Balanced metadata not available.", ln=True)
        
    pdf.section_title("Synthetic Data Summary")
    if balance_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(balance_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Balanced metadata not available.", ln=True)
    
    pdf.section_title("Train Test Split")
    if st.session_state.get("split_done", False):
        pdf.multi_cell(0, 10,
            "The balanced dataset was successfully split into training and testing sets.\n"
            "70% of the data was used for training.\n"
            "30% was used for evaluation.\n\n"
            "The split was performed after age-group balancing to ensure fair representation of all age groups in both sets."
        )
    else:
        pdf.multi_cell(0, 10, "Train-test split has not been completed or was not triggered during this session.")


    pdf.section_title("List of Model trained")
    selected_models = list(dict.fromkeys(st.session_state.get("selected_models", [])))

    if selected_models:
        model_list_str = ", ".join(selected_models)
        pdf.multi_cell(0, 10, f"{model_list_str} model{'s' if len(selected_models) > 1 else ''} were selected for training.")
    else:
        pdf.multi_cell(0, 10, "No models were selected for training.") 
    

    # 7. Evaluation on Balanced Data
    pdf.section_title("Evaluation on Balanced Dataset")
    eval_df = st.session_state.get("eval_df", None)

    if eval_df is not None and not eval_df.empty:
        pdf.multi_cell(0, 10, "The following table shows model evaluation results (AUC, pAUC, EER) on the balanced test dataset:")
        draw_table_paginated(pdf, eval_df)
    else:
        pdf.multi_cell(0, 10, "No evaluation results available. Please run evaluation after model training.")

    # 8. Age Group–wise Evaluation
    pdf.section_title("Age-Specific Evaluation")
    age_eval_df = st.session_state.get("age_eval_df", None)

    if age_eval_df is not None and not age_eval_df.empty:
        pdf.multi_cell(0, 10, "The following table presents the model evaluation results grouped by age group and dataset source.\nMetrics such as AUC, pAUC, and EER are reported per age group:")
        draw_table_paginated(pdf, age_eval_df)
    else:
        pdf.multi_cell(0, 10, "Age-specific evaluation has not been completed or no results were generated.")

    # 9. Source-based Training and Evaluation
    pdf.section_title("Source-Based Evaluation")
    cross_eval_df = st.session_state.get("cross_eval_df", None)

    if cross_eval_df is not None and not cross_eval_df.empty:
        pdf.multi_cell(0, 10,
            "The following table shows model performance when trained on one dataset source "
            "and evaluated on another. This cross-dataset evaluation helps assess how well models "
            "generalize across different data distributions.")
        draw_table_paginated(pdf, cross_eval_df)
    else:
        pdf.multi_cell(0, 10, "Source-based evaluation results are not available. Please run the evaluation after model training.")
    # 10. Grad-CAM Visualizations
    pdf.section_title("Grad-CAM Interpretability Visualizations")
    pdf.multi_cell(0, 10,
        "Grad-CAM visualizations highlight the regions of the input images that most strongly influenced the deepfake detection model's decisions. "
        "Below are examples of such heatmaps for real and fake frames:")

    def add_gradcam_images_to_pdf(images, label_title):
        if not images:
            pdf.multi_cell(0, 10, f"No {label_title.lower()} Grad-CAM images available.")
            return

        pdf.section_title(f"{label_title} Samples")

        image_width = 60  # mm
        image_height = 45
        spacing = 5
        margin_x = pdf.l_margin
        x = margin_x
        y = pdf.get_y()

        for i, img in enumerate(images):
            # Convert and save temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                Image.fromarray(img).save(tmpfile.name)
                tmp_path = tmpfile.name

            # Draw image
            pdf.image(tmp_path, x=x, y=y, w=image_width, h=image_height)
            x += image_width + spacing

            if (i + 1) % 3 == 0:
                x = margin_x
                y += image_height + spacing
                pdf.set_y(y)

            os.remove(tmp_path)
    # Add both sets
    add_gradcam_images_to_pdf(
        st.session_state.get("gradcam_images_real", []),
        label_title="Real"
    )

    add_gradcam_images_to_pdf(
        st.session_state.get("gradcam_images_fake", []),
        label_title="Fake"
    )
    pdf.add_page()
    #Help and Disclaimer Informations
    pdf.section_title("Help & Instructions")
    pdf.multi_cell(0, 10,
        "This application builds and evaluates an age-diverse deepfake detection pipeline:\n"
        "1. Import Videos (celeb-DF, FaceForensics++)\n"
        "2. Preprocessing\n"
        "3. Frame Extraction\n"
        "4. Frame Preview\n"
        "5. Age Annotation\n"
        "6. Visualization\n"
        "7. Dataset Balancing\n"
        "8. Train-Test Split\n"
        "9. Model Training\n"
        "10. Evaluation\n"
        "11. SimSwap Deepfake Generation\n"
        "12. Grad-CAM\n"
        "13. PDF Reporting\n"
        "14. Dataset Export"
    )

    # Security Disclaimer
    pdf.section_title("Security Disclaimer")
    pdf.multi_cell(0, 10,
        "- Public datasets only: celeb-DF-v2, FaceForensics++\n"
        "- Local processing only: no video files are uploaded or shared externally\n"
        "- This tool is intended for educational and research purposes only"
    )

    # Data Availability Disclaimer
    pdf.section_title("Data Availability Disclaimer")
    pdf.multi_cell(0, 10,
        "- Raw datasets are NOT downloadable via this tool\n"
        "- Metadata such as paths, labels, and age group annotations can be exported\n"
        "- Original datasets can be obtained from:\n"
        "  UTKFace: https://susanqq.github.io/UTKFace/ \n"
        "  Celeb-DF-v2: https://github.com/DigitalTrustLab/celeb-DF \n"
        "  FaceForensics++: https://github.com/ondyari/FaceForensics"
    )
    
    pdf.add_page()
    # Contact Details
    pdf.section_title("Thank You")
    pdf.multi_cell(0, 10,
        "I appreciate your time exploring the Age-Diverse Deepfake Dataset Builder!\n"
        "This tool was designed to support research and experimentation in fairness-aware deepfake detection.\n\n"
        "Contact: unishajoshi@email.com\n"
        "Institution: Department of Data Science, Grand Canyon University\n\n"
        "Feel free to reach out for improvements, collaboration, or to share your findings!"
    )
    
    pdf.output(output_path) 
    return output_path 