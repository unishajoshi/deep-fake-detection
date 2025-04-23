from fpdf import FPDF
import pandas as pd
import os
import cv2

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", "DejaVuLGCSans.ttf", uni=True)
        self.add_font("DejaVu", "B", "DejaVuLGCSans-Bold.ttf", uni=True)
        self.set_font("DejaVu", "", 11)
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Deepfake Dataset Pipeline Report", ln=True, align="C")
        self.ln(5)

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

def create_full_pdf_report(
    output_path="final_output/final_report.pdf",
    dataset_summary=None,
    cleaning_notes=None,
    frame_count_summary=None,
    sample_frame_paths=[],
    age_annotation_summary=None,
    balance_summary=None,
    eval_balanced=None,
    eval_agewise=None,
    eval_celeb=None,
    eval_ffpp=None,
    gradcam_paths=[]
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Dataset Summary
    pdf.section_title("1. Dataset Summary")
    if dataset_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(dataset_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Dataset summary not available.", ln=True)

    # 2. Cleaning Summary
    pdf.section_title("2. Cleaning Summary")
    pdf.multi_cell(0, 10, cleaning_notes or "No cleaning performed.")

    # 3. Frame Extraction
    pdf.section_title("3. Frame Extraction")
    pdf.multi_cell(0, 10, str(frame_count_summary) if frame_count_summary is not None else "No frame extraction summary available.")


    # 4. Sample Frames
    if sample_frame_paths:
        pdf.add_image_grid(sample_frame_paths, title="4. Sample Extracted Frames", grid=(2, 3))
    else:
        pdf.cell(0, 10, "No sample frames available.", ln=True)

    # 5. Age Annotation Summary
    pdf.section_title("5. Age Annotation Summary")
    if age_annotation_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(age_annotation_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Age annotation not available.", ln=True)

    # 6. Balance Summary
    pdf.section_title("6. Balanced Dataset Summary")
    if balance_summary is not None:
        pdf.add_table(pd.DataFrame.from_dict(balance_summary, orient="index").reset_index())
    else:
        pdf.cell(0, 10, "Balanced metadata not available.", ln=True)

    # 7. Evaluation on Balanced Data
    pdf.section_title("7. Evaluation on Balanced Dataset")
    if eval_balanced is not None:
        pdf.add_table(eval_balanced)
    else:
        pdf.cell(0, 10, "Balanced evaluation results not available.", ln=True)

    # 8. Age Group–wise Evaluation
    pdf.section_title("8. Age-Specific Evaluation")
    if eval_agewise is not None:
        pdf.add_table(eval_agewise)
    else:
        pdf.cell(0, 10, "Age-specific evaluation results not available.", ln=True)

    # 9. Source-based Training and Evaluation
    pdf.section_title("9. Source-Based Evaluation")
    if eval_celeb is not None:
        pdf.cell(0, 10, "celeb -> Balanced", ln=True)
        pdf.add_table(eval_celeb)
    else:
        pdf.cell(0, 10, "celeb results not available.", ln=True)
    
    pdf.ln(5)
    
    if eval_ffpp is not None:
        pdf.cell(0, 10, "FaceForensics++ -> Balanced", ln=True)
        pdf.add_table(eval_ffpp)
    else:
        pdf.cell(0, 10, "FaceForensics++ results not available.", ln=True)

    # 10. Grad-CAM Visualizations
    if gradcam_paths:
        pdf.add_image_grid(gradcam_paths, title="10. Grad-CAM Interpretability", grid=(2, 3))
    else:
        pdf.section_title("10. Grad-CAM")
        pdf.cell(0, 10, "No Grad-CAM visualizations found.", ln=True)

    pdf.output(output_path)
    return output_path