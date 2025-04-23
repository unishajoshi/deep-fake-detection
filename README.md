# Age-Diverse Deepfake Detection System
The Age-Diverse Deepfake Detection System is a modular
pipeline designed for creating an age balanced dataset to enhance age fairness in
the deepfake detection. The system automates the process of dataset
preparation, model training, evaluation, result visualization and exporting
results with a strong emphasis on age balance and fairness
in AI. This system consists of a simple
and intuitive Streamlit-based web interface, making it usable even by non-programmers or first-time researchers.

## Features
This Streamlit app helps researchers create and analyze a deepfake dataset balanced across age groups. It supports video import, age annotation, model training, evaluation, and PDF report generation. The key features of this application are was follows:

- Upload videos from different sources (e.g., Celeb-DF, FaceForensics++)
- Frame-level extraction 
- Age annotation using DeepFace
- Dataset cleaning, preprocessing, and age group balancing
- Model training:
  - **XceptionNet**, **EfficientNet**, **LipForensics**
- Evaluation metrics: AUC, pAUC, EER (overall + per age group)
- Model training and evaluation using source data for comparative analysis
- Grad-CAM visualization for interpretability
- Export detailed PDF report and ZIP of dataset outputs

## System Requirement
```bash
- OS: Windows, macOS, or Linux
- Memory: Minimum 16 GB RAM (32 GB+ recommended for large video datasets)
- GPU: NVIDIA GPU with CUDA support (recommended for model training and Grad-CAM)
- python: Python 3.8+
- git and ffmpeg
```

## Installation and Required Dependencies

Clone the repository and install dependencies:
```bash
git clone https://github.com/unishajoshi/deep-fake-detection.git
cd deep-fake-detection
pip install -r requirements.txt
```

## Execute the command

Run the Streamlit app:
```bash
python -m streamlit run main.py
```

Then open your browser to:
```
http://localhost:8501/
```

## Sample Dataset

You can request public datasets from:
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)

Filter videos using the metadata provided in this app.

## Acknowledgements

Developed by **Unisha Joshi** as part of a Capstone project.

---

**Note**: This app does not distribute deepfake video content. It uses publicly available datasets for academic purposes only.
