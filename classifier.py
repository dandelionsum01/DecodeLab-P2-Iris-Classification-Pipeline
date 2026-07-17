"""
classifier.py — Multi-Algorithm Iris Classification Pipeline
DecodeLabs AI Internship | Project 2

Algorithms  : KNN (auto-tuned K) · SVM · Random Forest · Logistic Regression
Outputs     : Terminal report + 5 PNG plots saved to ./plots/
Requirements: pip install scikit-learn matplotlib seaborn pandas numpy
Usage       : python classifier.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, label_binarize
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
PLOTS_DIR    = "plots"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
K_RANGE      = range(1, 31)

RST  = "\033[0m";  BOLD = "\033[1m"
CYN  = "\033[96m"; GRN  = "\033[92m"
YLW  = "\033[93m"

os.makedirs(PLOTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 1.  DATA
# ═══════════════════════════════════════════════════════════════════
def load_data():
    """Load Iris, return DataFrame + raw arrays."""
    iris = load_iris()
    df   = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.data, iris.target, iris.target_names, iris.feature_names


def print_eda(df, target_names):
    """Print dataset stats and class distribution to terminal."""
    print(f"\n{CYN}{BOLD}{'─'*62}")
    print("  DATASET OVERVIEW")
    print(f"{'─'*62}{RST}")
    print(f"  Samples  : {len(df)}")
    print(f"  Features : {len(df.columns) - 1}")
    print(f"  Classes  : {', '.join(target_names)}\n")
    print(f"{CYN}  Class distribution:{RST}")
    for name, cnt in df["species"].value_counts().sort_index().items():
        bar = "█" * cnt
        print(f"    {name:<14} {cnt:>3}  {bar}")
    print(f"\n{CYN}  Feature statistics:{RST}")
    print(df.drop("species", axis=1).describe().round(3).to_string())


# ═══════════════════════════════════════════════════════════════════
# 2.  PLOTS
# ═══════════════════════════════════════════════════════════════════
def _save(fig, fname):
    """Save figure and close."""
    path = os.path.join(PLOTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GRN}✓{RST} {path}")


def plot_pairplot(df):
    """Seaborn pairplot — feature relationships coloured by species."""
    print(f"  {YLW}→{RST} Pairplot …")
    sns.set_theme(style="ticks", palette="Set2")
    g = sns.pairplot(df, hue="species", diag_kind="kde",
                     plot_kws={"alpha": 0.55})
    g.fig.suptitle("Iris — Feature Pair Plot", y=1.02,
                   fontsize=13, fontweight="bold")
    path = os.path.join(PLOTS_DIR, "01_pairplot.png")
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {GRN}✓{RST} {path}")


def plot_elbow(X_tr, y_tr):
    """Cross-validated F1 vs K — pick the elbow."""
    print(f"  {YLW}→{RST} Elbow curve …")
    scores = [
        cross_val_score(KNeighborsClassifier(n_neighbors=k),
                        X_tr, y_tr, cv=5, scoring="f1_macro").mean()
        for k in K_RANGE
    ]
    best_k = list(K_RANGE)[int(np.argmax(scores))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_RANGE, scores, "o-", color="#4C72B0", lw=2, ms=6)
    ax.axvline(best_k, color="#DD8452", ls="--", lw=1.8,
               label=f"Optimal K = {best_k}")
    ax.set(xlabel="K", ylabel="CV F1-macro",
           title="KNN Elbow Curve — Finding Optimal K")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    _save(fig, "02_elbow_curve.png")
    print(f"  {GRN}→ Optimal K = {best_k}{RST}")
    return best_k


def plot_confusion_matrices(results, class_names):
    """2×2 grid of annotated confusion-matrix heatmaps."""
    print(f"  {YLW}→{RST} Confusion matrices …")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, r) in zip(axes.flatten(), results.items()):
        sns.heatmap(r["cm"], annot=True, fmt="d", ax=ax, cmap="Blues",
                    linewidths=0.5,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{name}\nAcc {r['accuracy']:.3f}  ·  F1 {r['f1']:.3f}",
                     fontweight="bold")
        ax.set(xlabel="Predicted", ylabel="Actual")
    fig.suptitle("Confusion Matrices — All Models",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "03_confusion_matrices.png")


def plot_roc_curves(results, X_te, y_te, n_classes):
    """One-vs-Rest ROC curves + AUC for all models."""
    print(f"  {YLW}→{RST} ROC curves …")
    y_bin  = label_binarize(y_te, classes=list(range(n_classes)))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, r) in zip(axes.flatten(), results.items()):
        proba = r["model"].predict_proba(X_te)
        for cls, col in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_bin[:, cls], proba[:, cls])
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"Class {cls}  AUC={auc(fpr, tpr):.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlim=[0, 1], ylim=[0, 1.05],
               xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"{name} — ROC")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "04_roc_curves.png")


def plot_comparison(results):
    """Grouped bar chart — accuracy / F1 / precision / recall per model."""
    print(f"  {YLW}→{RST} Model comparison chart …")
    names   = list(results.keys())
    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = ["Accuracy", "F1", "Precision", "Recall"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    x, w    = np.arange(len(names)), 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=lbl, color=col, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.set(ylabel="Score", ylim=[0, 1.15],
           title="Model Comparison — All Metrics")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "05_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# 3.  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════
def train_all(models, X_tr, X_te, y_tr, y_te):
    """Fit every model and collect all metrics in one dict."""
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_te)
        results[name] = {
            "model":     mdl,
            "preds":     preds,
            "cm":        confusion_matrix(y_te, preds),
            "accuracy":  accuracy_score(y_te, preds),
            "f1":        f1_score(y_te, preds, average="macro"),
            "precision": precision_score(y_te, preds, average="macro"),
            "recall":    recall_score(y_te, preds, average="macro"),
        }
    return results


def print_report(results, y_te, target_names):
    """Print formatted comparison table + full report for best model."""
    print(f"\n{CYN}{BOLD}{'─'*62}")
    print("  PERFORMANCE REPORT")
    print(f"{'─'*62}{RST}")
    hdr = f"  {'Model':<24} {'Accuracy':>9} {'F1':>7} {'Precision':>10} {'Recall':>8}"
    print(f"{BOLD}{hdr}{RST}")
    print(f"  {'─'*59}")
    best = max(results, key=lambda n: results[n]["f1"])
    for name, r in results.items():
        tag = f"  {GRN}← best{RST}" if name == best else ""
        print(f"  {name:<24} {r['accuracy']:>9.4f} {r['f1']:>7.4f}"
              f" {r['precision']:>10.4f} {r['recall']:>8.4f}{tag}")
    print(f"\n{CYN}{BOLD}  Best overall: {best}{RST}")
    print(f"\n{CYN}  Full classification report — {best}:{RST}")
    print(classification_report(y_te, results[best]["preds"],
                                 target_names=target_names))


# ═══════════════════════════════════════════════════════════════════
# 4.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print(f"\n{CYN}{BOLD}{'═'*62}")
    print("  IRIS CLASSIFICATION PIPELINE  ·  DecodeLabs P2")
    print(f"{'═'*62}{RST}")

    # ── 1. Load & EDA ──────────────────────────────────────────────
    print(f"\n{YLW}[1/5] Loading dataset …{RST}")
    df, X, y, tnames, _ = load_data()
    print_eda(df, tnames)

    # ── 2. EDA plot ────────────────────────────────────────────────
    print(f"\n{YLW}[2/5] EDA visualisation …{RST}")
    plot_pairplot(df)

    # ── 3. Preprocess ──────────────────────────────────────────────
    print(f"\n{YLW}[3/5] Preprocessing …{RST}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    print(f"  Train : {len(X_tr)} samples")
    print(f"  Test  : {len(X_te)} samples")
    print(f"  Scaler: StandardScaler  (μ=0, σ²=1)")
    best_k = plot_elbow(X_tr, y_tr)

    # ── 4. Train ───────────────────────────────────────────────────
    print(f"\n{YLW}[4/5] Training all models …{RST}")
    models = {
        f"KNN  (K={best_k})":    KNeighborsClassifier(n_neighbors=best_k),
        "SVM  (RBF)":           SVC(kernel="rbf", probability=True,
                                    random_state=RANDOM_STATE),
        "Random Forest":        RandomForestClassifier(n_estimators=100,
                                                       random_state=RANDOM_STATE),
        "Logistic Regression":  LogisticRegression(max_iter=1000,
                                                   random_state=RANDOM_STATE),
    }
    results = train_all(models, X_tr, X_te, y_tr, y_te)
    print_report(results, y_te, tnames)

    # ── 5. Visualise ───────────────────────────────────────────────
    print(f"\n{YLW}[5/5] Generating plots …{RST}")
    plot_confusion_matrices(results, tnames)
    plot_roc_curves(results, X_te, y_te, n_classes=len(tnames))
    plot_comparison(results)

    print(f"\n{GRN}{BOLD}{'═'*62}")
    print(f"  ✓  Done — plots saved to ./{PLOTS_DIR}/")
    print(f"{'═'*62}{RST}\n")


if __name__ == "__main__":
    main()
