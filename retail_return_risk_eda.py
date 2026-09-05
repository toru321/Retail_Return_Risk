import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging for audit trails and debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset with basic path validation and error handling."""
    path = Path(file_path)
    if not path.is_file():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The specified dataset was not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset successfully loaded from '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV file: {str(e)}")
        raise


def calculate_data_quality_and_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates summary statistics, missing values, duplicate counts, skewness,

    and correlation matrix for all numerical features.
    """
    logging.info("Calculating summary statistics and data quality metrics...")

    # Identify numerical columns for statistical computations
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        raise ValueError("No numerical features found in dataset for calculations.")

    # 1. Missing and Duplicate Audit
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    logging.info(f"Duplicate rows detected: {duplicate_rows} ({duplicate_rows / total_rows:.2%})")

    # 2. Comprehensive Numerical Summary Table
    summary_df = pd.DataFrame(
        {
            "Data Type": df[num_cols].dtypes,
            "Missing Count": df[num_cols].isnull().sum(),
            "Missing %": (df[num_cols].isnull().sum() / total_rows) * 100,
            "Mean": df[num_cols].mean(),
            "Median": df[num_cols].median(),
            "Std Dev": df[num_cols].std(),
            "Min": df[num_cols].min(),
            "Max": df[num_cols].max(),
            "Skewness": df[num_cols].skew(),
        }
    )

    # 3. Correlation Matrix
    corr_matrix = df[num_cols].corr()

    return summary_df, corr_matrix


def generate_eda_plots(
    df: pd.DataFrame,
    num_cols: Optional[list] = None,
    save_fig_prefix: Optional[str] = "eda",
) -> None:
    """Generates and displays Histograms, Boxplots, and a Correlation Heatmap.

    Optimized for layout clarity and performance.
    """
    logging.info("Generating EDA Visualizations...")

    # Filter numerical columns if not provided
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        logging.warning("No numerical features available for plotting.")
        return

    # Set visualization theme
    sns.set_theme(style="whitegrid", palette="muted")
    n_features = len(num_cols)

    # --- 1. Histograms (Feature Distributions) ---
    fig, axes = plt.subplots(
        nrows=(n_features + 2) // 3,
        ncols=min(3, n_features),
        figsize=(15, 3 * ((n_features + 2) // 3)),
    )
    axes = np.array(axes).reshape(-1)  # Flatten for uniform indexing

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    # Hide unused subplot axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_fig_prefix:
        plt.savefig(f"{save_fig_prefix}_histograms.png", dpi=300)
    plt.show()

    # --- 2. Boxplots (Outlier Analysis) ---
    fig, axes = plt.subplots(
        nrows=(n_features + 2) // 3,
        ncols=min(3, n_features),
        figsize=(15, 3 * ((n_features + 2) // 3)),
    )
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="lightsalmon")
        axes[i].set_title(f"Boxplot of {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_fig_prefix:
        plt.savefig(f"{save_fig_prefix}_boxplots.png", dpi=300)
    plt.show()

    # --- 3. Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()

    # Create a mask for the upper triangle to avoid redundant information
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    if save_fig_prefix:
        plt.savefig(f"{save_fig_prefix}_correlation_heatmap.png", dpi=300)
    plt.show()


# --- Sample Execution & Demonstration ---
if __name__ == "__main__":
    # Path to your project CSV
    CSV_FILE_PATH = "dataset_10_retail_return_risk.csv"

    try:
        # Load dataset
        df = load_dataset(CSV_FILE_PATH)
    except FileNotFoundError:
        logging.warning("Specified CSV not found. Generating mock retail return data for testing...")

        # Synthetic dataset generation for demonstration
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame(
            {
                "purchase_amount": np.random.exponential(scale=100, size=n_samples),
                "customer_age": np.random.normal(loc=38, scale=12, size=n_samples),
                "return_delay_days": np.random.poisson(lam=5, size=n_samples),
                "risk_score": np.random.uniform(low=0.0, high=1.0, size=n_samples),
                "target": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
            }
        )
        # Introduce synthetic missing values and duplicates for testing
        df.loc[10:20, "customer_age"] = np.nan
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    # Execute metrics extraction
    summary_stats, correlation = calculate_data_quality_and_stats(df)

    print("\n" + "=" * 80)
    print("DATA QUALITY & STATISTICAL SUMMARY")
    print("=" * 80)
    print(summary_stats.round(3).to_string())

    print("\n" + "=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)
    print(correlation.round(3).to_string())

    # Execute Visualizations
    generate_eda_plots(df, save_fig_prefix="retail_risk_eda")