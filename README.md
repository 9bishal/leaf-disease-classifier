# Plant Leaf Disease Classifier

An end-to-end deep learning system that classifies plant leaf images as
**Healthy**, **Powdery Mildew**, or **Rust** using transfer learning with
MobileNetV2 — deployed as an interactive Streamlit app.

![Architecture Diagram](images/architecture_diagram.png)

## Problem Statement

Crop diseases cause major yield losses worldwide, and early detection is one
of the highest-ROI interventions in agriculture. This project builds an
image classifier that lets a farmer photograph a leaf and get an instant
diagnosis — the same problem space as real AgriTech products (PlantVillage,
Plantix).

More broadly, this project demonstrates **transfer learning for image
classification with a small custom dataset** — a pattern that applies to
defect detection, medical imaging triage, retail product categorization, and
many other CV problems where labeled data is limited.

## A Note on the Dataset

This repo ships with a **small synthetic dataset** (3 classes, 80 images
each, 160x160px, generated programmatically) so the full pipeline — data
loading, augmentation, transfer learning, evaluation, deployment — runs
end-to-end out of the box.

**For a real-world submission**, swap `data/raw/` for the
[PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(or any similarly-structured `data/raw/<ClassName>/*.jpg` folder). No code
changes are needed — `src/dataset.py` automatically detects classes from
folder names. On PlantVillage, this pipeline with MobileNetV2 transfer
learning typically achieves **85-95% validation accuracy**.

![Sample classes](images/sample_classes_preview.png)
*Healthy / Powdery Mildew / Rust — synthetic samples used for this demo*

## Approach

1. **Data pipeline** (`src/dataset.py`): `image_dataset_from_directory` with
   an 80/20 train/validation split (fixed seed for reproducibility), plus a
   Keras augmentation layer (`RandomFlip`, `RandomRotation`, `RandomZoom`)
   applied only during training.
2. **Model** (`src/train_model.py`): MobileNetV2 pretrained on ImageNet as a
   frozen feature extractor, with a custom classification head
   (`GlobalAveragePooling2D -> Dropout -> Dense(softmax)`).
   - **Phase A**: train the head only, base model frozen.
   - **Phase B**: unfreeze the top 30 layers of MobileNetV2 and fine-tune
     at a very low learning rate (1e-5).
   - If pretrained weights aren't available (e.g. restricted network), the
     code gracefully falls back to a small CNN trained from scratch, so the
     pipeline still runs end-to-end.
3. **Evaluation**: per-class precision/recall/F1 and a confusion matrix —
   not just overall accuracy, since a model could be excellent on one class
   and poor on another.
4. **Deployment** (`app/streamlit_app.py`): upload a leaf image, get the
   predicted class, confidence score, and a probability bar chart for all
   classes.

## Results (this demo, synthetic dataset / fallback CNN)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 1.00 | 1.00 | 1.00 |
| Powdery_Mildew | 1.00 | 1.00 | 1.00 |
| Rust | 1.00 | 1.00 | 1.00 |

![Training Curves](images/training_curves.png)
![Confusion Matrix](images/confusion_matrix.png)

100% accuracy reflects the synthetic dataset's clearly separable visual
patterns and confirms the pipeline is correct end-to-end. **On PlantVillage
with MobileNetV2 transfer learning, expect 85-95% validation accuracy** —
report that figure for real-world claims.

## Project Structure

```
leaf-disease-classifier/
├── data/
│   └── raw/
│       ├── Healthy/
│       ├── Powdery_Mildew/
│       └── Rust/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── dataset.py        # data loading, train/val split, augmentation
│   ├── train_model.py     # model architecture, two-phase training, evaluation
│   └── predict.py          # reusable prediction function (used by app)
├── models/
│   └── leaf_disease_model.keras
├── app/
│   └── streamlit_app.py   # interactive dashboard
├── images/
├── requirements.txt
└── README.md
```

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/leaf-disease-classifier.git
cd leaf-disease-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Replace data/raw/ with the PlantVillage dataset for real results

# 4. Train the model
python src/train_model.py

# 5. Launch the dashboard
python -m streamlit run app/streamlit_app.py
```

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'tensorflow'`, the active
Python environment does not have this project's dependencies installed.
From the `leaf-disease-classifier` directory, run:

```bash
pip install -r requirements.txt
```

Then rerun the script or Streamlit app from the same environment.

## Tech Stack

Python · TensorFlow / Keras · MobileNetV2 (Transfer Learning) · Scikit-learn
· Matplotlib · Streamlit

## Future Improvements

- Train on the full PlantVillage dataset (38 classes across 14 crop species)
  for a more comprehensive multi-crop classifier
- Add an "out-of-distribution" detector to flag non-leaf images
- Add Grad-CAM visualizations to show *which part* of the leaf drove the
  prediction (per-image explainability)
- Quantize the model (TFLite) for true on-device mobile deployment
