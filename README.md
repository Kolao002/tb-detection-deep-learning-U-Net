# Tuberculosis Detection from Chest X-ray with U-Net Segmentation 

This project implements a complete deep learning pipeline for automated Tuberculosis (TB) detection from chest X-ray images using transfer learning, U-Net segmentation, Score-CAM visualization, and feature space analysis.

The project is based on the methodology proposed by Rahman et al. (IEEE Access, 2020) and focuses heavily on reducing dataset bias caused by multi-source image heterogeneity.

## Features

- Chest X-ray preprocessing pipeline
- Local Contrast Normalization (LCN)
- U-Net lung segmentation
- TB classification using:
  - CheXNet
  - DenseNet201
  - ResNet50
  - MobileNetV2
- Score-CAM explainability visualization
- t-SNE feature embedding analysis
- ROC curves and confusion matrix evaluation
- Strong regularization using:
  - AdamW
  - Label smoothing
  - Warmup cosine annealing
  - Gradient clipping

---

## Dataset

The project uses the publicly available Tuberculosis Chest X-Ray dataset assembled by Rahman et al.

Dataset sources include:
- NLM Montgomery County dataset
- Shenzhen dataset
- Belarus TB dataset
- NIAID TB Portal
- RSNA Pneumonia Detection Challenge dataset

Total images: **7,000**
- TB Positive: 3,500
- Normal: 3,500

---

## Preprocessing Pipeline

The preprocessing pipeline removes acquisition-specific artifacts and biases using:

1. Local Contrast Normalization
2. Resize and center crop
3. Per-image Z-score normalization

This ensures the model learns lung pathology rather than dataset-specific artifacts.

---

## Models Used

| Model | Accuracy | Sensitivity | AUC |
|------|------|------|------|
| CheXNet | 99.05% | 98.10% | 0.9989 |
| DenseNet201 | 99.21% | 100.0% | 0.9981 |
| ResNet50 | 98.89% | 99.05% | 0.9969 |
| MobileNetV2 | 98.10% | 92.38% | 0.9950 |

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Albumentations
- Grad-CAM / Score-CAM
- Jupyter Notebook

---

## Project Structure

```text
project/
│
├── Final_Code_Cleaned.ipynb
├── README.md
├── requirements.txt
├── dataset/
├── models/
├── outputs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── scorecam/
│   └── tsne/
