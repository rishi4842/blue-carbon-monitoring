from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURES = ["AGC", "SOCD", "NDVI", "GreenTrend", "MTemp", "TotPrcp"]


def perform_eda():
    project_dir = Path(__file__).resolve().parents[1]
    dataset_path = project_dir / "data" / "processed" / "final_dataset.csv"
    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ML dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing feature columns: {missing_features}")
    if df.empty:
        raise ValueError("Dataset contains no rows to analyze")

    df_features = df[FEATURES]
    if df_features.isna().any().any():
        raise ValueError("Feature data contains missing values; clean it before EDA")

    statistics = df_features.describe().T
    correlations = df_features.corr()
    constant_features = [
        feature for feature in FEATURES if df_features[feature].nunique() <= 1
    ]

    print(f"\nAnalyzing {len(df_features):,} pixels across {len(FEATURES)} features.")
    print("\n--- DATASET STATISTICS ---")
    print(statistics)
    if constant_features:
        print(
            "\nFeatures with no variation (correlations are undefined; "
            f"exclude from PCA): {', '.join(constant_features)}"
        )

    statistics.to_csv(reports_dir / "feature_statistics.csv")
    correlations.to_csv(reports_dir / "correlation_matrix.csv")

    print("\nGenerating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlations,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        mask=correlations.isna(),
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=20)
    if constant_features:
        fig.text(
            0.5,
            0.01,
            "Blank cells indicate undefined correlations for constant features.",
            ha="center",
            fontsize=9,
        )
    fig.savefig(
        figures_dir / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Generating feature distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Distributions of Environmental and Carbon Features", fontsize=16)

    for index, feature in enumerate(FEATURES):
        ax = axes[index // 3, index % 3]
        sns.histplot(
            df_features[feature],
            bins=50,
            ax=ax,
            color="teal",
            kde=feature not in constant_features,
        )
        ax.set_title(f"{feature} Distribution")
        ax.set_ylabel("Pixel Count")

    fig.tight_layout()
    fig.savefig(
        figures_dir / "feature_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nEDA complete. Figures saved to: {figures_dir}")
    print(f"Statistics saved to: {reports_dir / 'feature_statistics.csv'}")
    print(f"Correlations saved to: {reports_dir / 'correlation_matrix.csv'}")


if __name__ == "__main__":
    perform_eda()
