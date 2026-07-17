# Brain CT Hemorrhage Classification and Segmentation

A machine learning pipeline for the automated detection, classification, and pixel-level segmentation of cerebral hemorrhages from brain CT scans. This project explores a progression of modeling approaches — from baseline logistic regression to convolutional neural networks (CNNs) — and incorporates model interpretability via Grad-CAM++ and pixel-level localization via U-Net segmentation.

📄 [View Final Report](https://www.dropbox.com/scl/fi/nk1s7sfad851srp6ey1vs/MATH_7243_Final_Project-4.pdf?rlkey=cbon4qgjk5rjttifx0dm3x5h6&st=4gkwfeft&dl=0)

---

## Motivation

Cerebral hemorrhage is a life-threatening condition that requires rapid and accurate diagnosis. Even for experienced radiologists, the high levels of noise and visual similarity between hemorrhage types in CT scans make this a difficult task. This project investigates whether machine learning techniques can assist in automating hemorrhage detection and classification, with the goal of improving diagnostic speed and consistency.

---

## Dataset

The dataset consists of brain CT scans organized by hemorrhage type, stored in DICOM format. Images were sourced from the `brain_window` rendering subfolder for each class. The following hemorrhage types are represented:

- Epidural
- Intraparenchymal
- Intraventricular
- Subarachnoid
- Subdural
- Normal (no hemorrhage)
- Multi (multiple hemorrhage types)

Due to computational constraints, a stratified random sample of 500 images was used for baseline models, while the full dataset was used for CNN training.

> **Note:** The dataset is not included in this repository. To run the notebooks, download the dataset and update the `data_dir` variable in the first cell of each notebook to point to your local copy.

---

## Demo App — Sample Image Attributions

An interactive Streamlit app (`app/streamlit_app.py`) lets users run the CNN on either an uploaded scan or one of seven pre-loaded sample images — one per class. Since the original training data is proprietary and cannot be redistributed, these sample images are instead sourced from [Radiopaedia.org](https://radiopaedia.org) under the [CC BY-NC-SA 3.0 license](https://creativecommons.org/licenses/by-nc-sa/3.0/) and used here for non-commercial, educational purposes only.

| Class | Source |
|---|---|
| Epidural | Shif Y, Epidural hemorrhage. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-26277 |
| Intraparenchymal | Puyó D, Lobar intraparenchymal hemorrhage. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-20297 |
| Intraventricular | Knipe H, Primary intraventricular hemorrhage. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-53317 |
| Subarachnoid | Puyó D, Subarachnoid hemorrhage. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-22377 |
| Subdural | Gaillard F, Subdural hemorrhage. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-17559 |
| Multi | Sorrentino S, Combination of subdural, epidural, and subarachnoid hemorrhage in an open skull fracture. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-14868 |
| Normal | Bø H, Normal brain CT. Case study, Radiopaedia.org (Accessed on 17 Jul 2026) https://doi.org/10.53347/rID-230970 |

---

## Project Structure

```
brain-ct-hemorrhage-classification-segmentation/
│
├── models/
│   ├── logistic_regression/
│   │   └── baseline_models.ipynb       # Binary, multiclass softmax, and multilabel sigmoid logistic regression
│   ├── cnn/
│   │   ├── cnn_models.ipynb            # Sigmoid CNN, SoftMax CNN, AI-assisted CNN, and final trained model
│   │   ├── CNNmodel1.keras             # Initial SoftMax CNN (15 epochs)
│   │   ├── CNNImprovedModel.keras      # Continued training checkpoint (15 epochs)
│   │   └── CNNImprovedModel_best.keras # Best weights from final training run (20 epochs total)
│   └── unet/
│       ├── data_mask_for_unet.ipynb    # Preprocessing pipeline: generates binary and quad-level segmentation masks
│       └── unet.ipynb                  # U-Net model architecture, training, and evaluation
│
├── app/
│   ├── streamlit_app.py                # Interactive demo app (upload a scan or pick a sample image)
│   ├── requirements.txt
│   └── sample_images/                  # One demo image per class (see attributions above)
│
├── reports/
│   └── final_report.pdf                # Full project report
└── README.md
```

---

## Modeling Approaches

### Baseline Models (`models/logistic_regression/`)
Three logistic regression variants were implemented as baselines:

- **Binary logistic regression** — hemorrhage vs. normal; achieved ~93–95% test accuracy but cannot distinguish hemorrhage subtypes
- **Multiclass SoftMax logistic regression** — predicts a single hemorrhage type per image; achieved ~31% test accuracy due to loss of spatial information when images are flattened
- **Multilabel sigmoid logistic regression** — one-vs-rest approach allowing multiple class predictions; achieved ~14% test accuracy, reflecting a mismatch between the model formulation and the dataset's mutually exclusive label structure

### CNN Models (`models/cnn/`)
Four CNN variants were implemented and compared:

- **Sigmoid CNN** — multi-label output; showed clear overfitting (~75% training accuracy, ~25–30% validation accuracy)
- **SoftMax CNN (baseline)** — more appropriate for mutually exclusive classes; achieved ~57% training and validation accuracy
- **AI-assisted CNN** — incorporated class weighting, data augmentation, batch normalization, and global average pooling; ultimately underperformed with ~42% peak validation accuracy
- **Final SoftMax CNN** — continued training of the original SoftMax CNN with early stopping and ReduceLROnPlateau; achieved the best overall performance at ~59% validation accuracy after 20 epochs

### Grad-CAM++
Gradient-weighted Class Activation Mapping (Grad-CAM++) was applied to the sigmoid CNN to visualize which image regions influenced model predictions. Results highlighted that high accuracy alone does not guarantee that a model is attending to clinically meaningful regions.

> **Note:** Grad-CAM++ code will be added to this repository shortly.

### U-Net Segmentation (`models/unet/`)
A U-Net architecture was trained for pixel-level segmentation, classifying each pixel into one of four anatomical categories: background, brain tissue, skull, and hemorrhage region. The model achieved a training accuracy of ~80% and a Mean IoU of ~0.43, with no observed overfitting.

---

## Key Results

| Model | Test Accuracy |
|---|---|
| Binary Logistic Regression | ~93–95% |
| Multiclass SoftMax Logistic Regression | ~31% |
| Multilabel Sigmoid Logistic Regression | ~14% |
| Sigmoid CNN | ~25–30% (val) |
| SoftMax CNN (final, 20 epochs) | ~59% (val) |
| U-Net (pixel-level) | ~80% accuracy, ~0.43 Mean IoU |

---

## Requirements

```
numpy
matplotlib
Pillow
scikit-learn
tensorflow
keras
seaborn
```

Install dependencies with:
```bash
pip install numpy matplotlib Pillow scikit-learn tensorflow keras seaborn
```

---

## How to Run

1. Clone the repository
2. Download the dataset and update `data_dir` in the first cell of each notebook to point to your local copy
3. For classification: run `models/logistic_regression/baseline_models.ipynb` then `models/cnn/cnn_models.ipynb`
4. For segmentation: run `models/unet/data_mask_for_unet.ipynb` first to generate masks, then `models/unet/unet.ipynb`
5. Pre-trained model weights (`.keras` files) can be loaded directly to skip retraining

---

## Authors

This project was completed as the final project for MATH 7243: Machine Learning I at Northeastern University (Spring 2026).

| Contributor | Primary Contributions |
|---|---|
| Dylan Chern | CNN models, baseline logistic regression models, preprocessing |
| Nikki Budri | Preprocessing |
| Jiacong Ye | Grad-CAM++ |
| Xiaozhe Zhang | U-Net preprocessing and segmentation |

Course Instructor: Prof. He Wang

---

## References

- Ahmed, S. N., & Prakasam, P. (2025). Intracranial hemorrhage segmentation and classification framework in computer tomography images using deep learning techniques. *Scientific Reports*, 15(1), 17151.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *arXiv:1505.04597*.
- Yamauchi, T. (2024). Spatial sensitive Grad-CAM++. *CVPR Workshops*, pp. 8164–8168.
- Google. Image segmentation. TensorFlow. https://www.tensorflow.org/tutorials/images/segmentation
