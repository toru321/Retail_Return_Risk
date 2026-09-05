import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def evaluate_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
    save_prefix: str = "model_evaluation",
) -> pd.DataFrame:
    """Evaluates the relationship and correlation between actual and predicted targets,

    generating heatmaps, ROC curves, and quantitative metrics.
    """
    logging.info("Calculating correlation and classification metrics...")

    # Ensure inputs are standard numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Pearson Correlation between Actual and Predicted
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    # 2. Performance Metrics
    metrics = {
        "Metric": [
            "Pearson Correlation",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC Score",
        ],
        "Value": [
            correlation,
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        ],
    }
    metrics_df = pd.DataFrame(metrics)

    print("\n" + "=" * 60)
    print(" ACTUAL VS. PREDICTED EVALUATION METRICS")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    # --- VISUALIZATIONS ---
    sns.set_theme(style="whitegrid")

    if y_prob is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Confusion Matrix Heatmap (Actual vs Predicted Counts)
    cm = confusion_matrix(y_true, y_pred)
    cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Percentages

    # Format annotations: Count + (Percentage)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        ax=axes[0],
        cbar=False,
        linewidths=1,
    )
    axes[0].set_title("Confusion Matrix (Actual vs Predicted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("Actual Label", fontsize=11)
    axes[0].set_xticklabels(["0 (No Risk)", "1 (Return Risk)"])
    axes[0].set_yticklabels(["0 (No Risk)", "1 (Return Risk)"])

    # Plot 2: Correlation & Prediction Agreement Matrix
    eval_data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    corr_matrix = eval_data.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[1],
        linewidths=1,
    )
    axes[1].set_title(f"Correlation Matrix\n(r = {correlation:.3f})", fontsize=12, fontweight="bold")

    # Plot 3: Predicted Probability Density by Actual Target (If probabilities provided)
    if y_prob is not None:
        prob_df = pd.DataFrame({"Actual": y_true, "Probability": y_prob})
        sns.kdeplot(
            data=prob_df,
            x="Probability",
            hue="Actual",
            common_norm=False,
            fill=True,
            palette={0: "blue", 1: "red"},
            alpha=0.4,
            ax=axes[2],
        )
        axes[2].set_title("Predicted Probability Density by Actual Class", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Predicted Probability of Return Risk")
        axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_correlation.png", dpi=300)
    plt.show()

    return metrics_df


# --- Example Integration with Your Pipeline ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # 1. Synthetic setup (Replace with your actual X_train, X_test, y_train, y_test)
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # 2. Fit Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 3. Generate Predictions & Probabilities
    y_predictions = model.predict(X_test)
    y_probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # 4. Run Evaluation & Correlation Plot
    evaluate_actual_vs_predicted(
        y_true=y_test,
        y_pred=y_predictions,
        y_prob=y_probabilities,
        save_prefix="retail_return_risk",
    )