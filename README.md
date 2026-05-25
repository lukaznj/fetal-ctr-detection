# Fetal Cardiothoracic Ratio (CTR) Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated biomedical system designed for the segmentation and estimation of the **fetal Cardiothoracic Ratio (CTR)** from clinical ultrasound images in the four-chamber view. 

This project was developed as part of the *Digital Image Processing and Analysis* (DOAS) course at the **Faculty of Electrical Engineering and Computing (FER), University of Zagreb**. It implements and rigorously compares two distinct paradigms of computer vision:
1. **Classical Computer Vision Pipeline:** An engineered pipeline incorporating Bilateral filtering, CLAHE, dual-Canny edge detection, Morphological operators, Hough-like ellipse fitting, and a custom anatomical pairing heuristic.
2. **Deep Learning Approach (U-Net):** A customized, end-to-end semantic segmentation U-Net model trained with extensive spatial and photometric data augmentations to directly predict heart and thorax masks, followed by robust boundary-based ellipse fitting.

Both methodologies are evaluated on the publicly available, clinical **FOCUS dataset** containing ultrasound scans annotated with manual ellipses by expert sonographers.

---

## 📊 Performance Comparison: Classical vs. Deep Learning

The following table summarizes the quantitative results evaluated across all 50 test images in the FOCUS database:

| Metric | Classical CV | Deep Learning (U-Net) | Performance Improvement / Notes |
| :--- | :---: | :---: | :--- |
| **Successful CTR Detections** | 94.00% (47/50) | **100.0% (50/50)** | DL successfully processed all edge cases |
| **Mean Absolute Error (MAE)** | 0.2661 | **0.0584** | **4.6x** lower error (highly accurate) |
| **Median Absolute Error** | 0.2919 | **0.0449** | **6.5x** improvement in median case |
| **Error Standard Deviation** | 0.1709 | **0.0478** | **3.6x** greater consistency and stability |
| **Max Error** | 0.7835 | **0.1993** | **3.9x** lower worst-case error (regularization effect) |
| **Mean Relative Error (MRE)** | 48.94% | **11.00%** | Biometric accuracy jumped from 51.1% to **89.0%** |
| **Mean IoU (Heart)** | 0.096 | **0.730** | **7.6x** better spatial localization of ventricles |
| **Mean IoU (Thorax)** | 0.268 | **0.812** | **3.0x** better spatial localization of chest |
| **Udio slika unutar tolerancije $\pm 5\%$** | 10.60% (5/47) | **58.0% (29/50)** | **5.5x** more clinical-grade matches |
| **Udio slika unutar tolerancije $\pm 15\%$** | 25.50% (12/47) | **94.0% (47/50)** | **3.7x** more acceptable-grade matches |

---

## 🛠️ Repository Structure

```text
.
├── classical/            # Classical CV pipeline scripts (preprocessing, detection, evaluation)
├── deep_learning/        # U-Net architecture, training script, datasets, and models
│   ├── models/           # Pre-trained baseline and augmented weights (.keras)
│   └── evaluate.py       # Deep learning inference and metric computation script
├── dataset/              # FOCUS dataset files (split into training/validation/testing)
├── results/              # Auto-generated CSVs, logs, error distribution graphs, and visualizations
├── metrics.py            # Unified clinical and spatial metrics (CTR, centroid, axis, angle, Combined Error)
├── requirements.txt      # List of third-party python dependencies
└── README.md             # This file
```

---

## 💻 Environment Setup & Installation

Before running the evaluation scripts, set up a virtual environment and install the required dependencies:

**1. Create a virtual environment**
Open your terminal in the root directory and run:
```bash
python3 -m venv .venv
```

**2. Activate the virtual environment**
*   On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```
*   On Windows:
    ```cmd
    .venv\Scripts\activate
    ```

**3. Install dependencies**
With the virtual environment active, run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔍 Running the Pipelines

### 1. Classical CV Pipeline
The classical code resides in `classical/`. It filters speckle noise, extracts contours, and searches for thorax-heart ellipse candidates using a heuristic scoring function based on anatomical hierarchy.

Run from the root directory:
```bash
# Evaluate all test images and output accuracy analysis
python classical/evaluate.py

# Evaluate a single image and save the 3-panel visualization
python classical/evaluate.py --image 012 --visualize

# Evaluate and save visualizations for all images
python classical/evaluate.py --visualize
```

### 2. Deep Learning Pipeline (U-Net)
The deep learning code resides in `deep_learning/`. It leverages a customized U-Net model trained on annotated masks.

We provide two pre-trained models in `deep_learning/models/`:
*   **`unet_augmented.keras`** (Default) – Trained with heavy spatial (rotation, translation) and photometric (brightness, contrast) augmentations.
*   **`unet_baseline.keras`** – Trained on clean raw images without augmentations.

Run from the root directory:
```bash
# Evaluate all images using the augmented model
python deep_learning/evaluate.py

# Evaluate a single image and save the labeled visual
python deep_learning/evaluate.py --image 012 --visualize

# Evaluate and save visualizations for all images
python deep_learning/evaluate.py --visualize

# Evaluate using the baseline (non-augmented) model
python deep_learning/evaluate.py --model unet_baseline.keras
```

---

## 📈 Evaluation Metrics & Labeled Visualizations

To ensure a fair and scientifically rigorous comparison, both evaluation pipelines output identical metrics through `metrics.py`:

### 1. Cardiothoracic Ratio (CTR)
The CTR represents the ratio of the major axis of the cardiac ellipse to that of the thoracic ellipse:

$$
\text{CTR}_{\text{auto}} = \frac{a_{\text{cardiac}}}{a_{\text{thorax}}}
$$

*   **AbsErr (Absolute Error) / MAE:** The absolute distance between $\text{CTR}_{\text{auto}}$ and the clinician's manual ground truth $\text{CTR}_{\text{GT}}$.

### 2. Spatial Metrics (IoU)
**Intersection over Union (IoU)** measures the overlap between predicted masks/ellipses and expert annotations. While the final CTR is a scalar ratio (which can occasionally be correct by mere coincidence), **IoU is the ultimate metric** verifying that the model has actually located the *correct organs in the correct places*.

### 3. Combined Error Metric ($E$)
To prevent mathematical flukes, we define a comprehensive **Combined Error** ($E \in [0, 1]$) representing a weighted sum of five bounded geometric parameters (lower is better):

$$
E = 0.40 \cdot e_{\text{ctr}} + 0.25 \cdot (1 - \text{IoU}_{\text{mean}}) + 0.15 \cdot e_{\text{centroid}} + 0.10 \cdot e_{\text{axis}} + 0.10 \cdot e_{\text{angle}}
$$

Where:
*   **`e_ctr` (MRE, 40%):** Relative deviation of the calculated CTR (most important clinical measure).
*   **`IoU` (Overlap, 25%):** Average IoU of both organs. High IoU proves spatial accuracy.
*   **`e_centroid` (15%):** Distance between the centers of predicted and GT ellipses, normalized by organ radius.
*   **`e_axis` (10%):** Average relative error of the major ($a$) and minor ($b$) semi-axes.
*   **`e_angle` (10%):** Angular difference between the ellipse rotation angles, taking $180^\circ$ symmetry into account.

Every image is categorized into one of four qualitative bins based on $E$:
*   **Excellent:** $E \le 0.20$
*   **Good:** $0.20 < E \le 0.40$
*   **Poor:** $0.40 < E \le 0.60$
*   **Bad:** $E > 0.60$

All logs, CSV tables, and error histograms are systematically saved inside `results/`.
