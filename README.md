<p align="center">
  <video src="https://github.com/user-attachments/assets/3681e759-c5ef-4578-b3ef-2962106a10a3" width="100%" autoplay muted loop playsinline></video>
</p>

# Iris Classification Pipeline — DecodeLabs P2

Extended version  of DecodeLabs Project 2. Goes beyond the basic KNN requirement by running a full multi-algorithm comparison pipeline with automated tuning, rich visualisations, and a formatted terminal report.

## Algorithms

| Model | Description |
|---|---|
| KNN (auto-tuned) | K chosen automatically via 5-fold cross-validation elbow curve |
| SVM (RBF kernel) | Support Vector Machine with radial basis function kernel |
| Random Forest | 100-tree ensemble classifier |
| Logistic Regression | Linear probabilistic classifier |

## Results

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| KNN (K=5) | 0.9333 | 0.9327 | 0.9444 | 0.9333 |
| **SVM (RBF)** | **0.9667** | **0.9666** | **0.9697** | **0.9667** |
| Random Forest | 0.9000 | 0.8997 | 0.9024 | 0.9000 |
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

**Best model: SVM (RBF)** at 96.7% accuracy.

## Plots Generated

All saved automatically to `./plots/` on every run:

- `01_pairplot.png` — feature relationships coloured by species
- `02_elbow_curve.png` — cross-validated F1 vs K to find optimal K
- `03_confusion_matrices.png` — all 4 models side by side
- `04_roc_curves.png` — ROC curves + AUC per class per model
- `05_model_comparison.png` — grouped bar chart of all metrics

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run:
   ```
   python classifier.py
   ```

## How It Works

1. **Load** — Iris dataset loaded from scikit-learn (150 samples, 4 features, 3 classes)
2. **EDA** — class distribution and feature stats printed to terminal, pairplot saved
3. **Preprocess** — 80/20 stratified train/test split, StandardScaler applied (μ=0, σ²=1)
4. **Tune** — optimal K for KNN found via 5-fold cross-validation across K=1–30
5. **Train** — all 4 models trained on scaled training set
6. **Evaluate** — accuracy, F1, precision, recall computed; full classification report for best model
7. **Visualise** — confusion matrices, ROC curves, model comparison chart saved as PNGs

## Project Structure

```
w2/
├── classifier.py       # main pipeline
├── requirements.txt    # dependencies
├── README.md           # this file
└── plots/              # auto-created on run
    ├── 01_pairplot.png
    ├── 02_elbow_curve.png
    ├── 03_confusion_matrices.png
    ├── 04_roc_curves.png
    └── 05_model_comparison.png
```
