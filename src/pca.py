from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURES = ["AGC", "SOCD", "NDVI", "MTemp", "TotPrcp"]


def perform_pca():
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = processed_dir / "final_dataset.csv"
    df = pd.read_csv(dataset_path)

    required_columns = ["Latitude", "Longitude", *FEATURES]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    if df.empty:
        raise ValueError("Dataset contains no rows for PCA")

    X = df[FEATURES]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("PCA features contain missing or non-finite values")

    constant_features = [name for name in FEATURES if X[name].nunique() <= 1]
    if constant_features:
        raise ValueError(
            "PCA features must have non-zero variance; constant columns found: "
            f"{constant_features}"
        )

    print("Using features:", ", ".join(FEATURES))
    print("Scaling features using StandardScaler...")
    X_scaled = StandardScaler().fit_transform(X)

    print("Fitting PCA model...")
    pca = PCA()
    pca_features = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)

    print("\n--- PCA VARIANCE EXPLAINED ---")
    for index, (variance, cumulative) in enumerate(
        zip(explained_variance, cumulative_variance), start=1
    ):
        print(f"PC{index}: {variance:.2f}% (Cumulative: {cumulative:.2f}%)")

    component_names = [f"PC{index}" for index in range(1, len(FEATURES) + 1)]
    pca_df = df[["Latitude", "Longitude"]].copy()
    pca_df[component_names] = pca_features
    pca_path = processed_dir / "pca_dataset.csv"
    pca_df.to_csv(pca_path, index=False)
    print(f"\nPCA dataset saved to: {pca_path}")

    variance_df = pd.DataFrame(
        {
            "ExplainedVariancePercent": explained_variance,
            "CumulativeVariancePercent": cumulative_variance,
        },
        index=component_names,
    )
    variance_df.index.name = "Component"
    variance_df.to_csv(reports_dir / "pca_explained_variance.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(explained_variance) + 1)
    ax.bar(
        components,
        explained_variance,
        alpha=0.7,
        color="teal",
        label="Individual variance",
    )
    ax.plot(
        components,
        cumulative_variance,
        marker="o",
        linestyle="-",
        color="orange",
        linewidth=2,
        label="Cumulative variance",
    )
    ax.set_title("PCA Explained Variance (Scree Plot)", fontsize=14, pad=15)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xticks(components)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    scree_path = figures_dir / "pca_variance.png"
    fig.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=FEATURES,
    )
    loadings.index.name = "Feature"
    loadings_path = reports_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)

    print("\n--- PCA LOADINGS ---")
    print(loadings.round(3))
    print(f"\nScree plot saved to: {scree_path}")
    print(f"Loadings saved to: {loadings_path}")


if __name__ == "__main__":
    perform_pca()
