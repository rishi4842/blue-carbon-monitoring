from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PCA_FEATURES = ["PC1", "PC2", "PC3", "PC4"]
RAW_FEATURES = ["AGC", "SOCD", "NDVI", "MTemp", "TotPrcp"]


def perform_clustering():
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"
    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pca_path = processed_dir / "pca_dataset.csv"
    original_path = processed_dir / "final_dataset.csv"
    print(f"Loading PCA dataset from {pca_path}...")
    df_pca = pd.read_csv(pca_path)
    df_original = pd.read_csv(original_path)

    missing_pca = [
        column
        for column in ["Latitude", "Longitude", *PCA_FEATURES]
        if column not in df_pca.columns
    ]
    missing_original = [
        column
        for column in ["Latitude", "Longitude", *RAW_FEATURES]
        if column not in df_original.columns
    ]
    if missing_pca:
        raise ValueError(f"PCA dataset is missing columns: {missing_pca}")
    if missing_original:
        raise ValueError(f"Original dataset is missing columns: {missing_original}")
    if len(df_pca) != len(df_original):
        raise ValueError(
            "PCA and original datasets have different row counts; "
            "cannot safely associate clusters with source pixels"
        )
    if not np.array_equal(
        df_pca[["Latitude", "Longitude"]].to_numpy(),
        df_original[["Latitude", "Longitude"]].to_numpy(),
    ):
        raise ValueError(
            "PCA and original datasets have different coordinate ordering; "
            "cannot safely associate clusters with source pixels"
        )

    X = df_pca[PCA_FEATURES]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("PCA components contain missing or non-finite values")
    if len(X) < 3:
        raise ValueError("At least 3 samples are required to evaluate clustering")

    max_k = min(8, len(X) - 1)
    print(f"\nCalculating silhouette scores for K=2 through K={max_k}...")
    scores = []
    best_k = None
    best_score = -1.0
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        cluster_count = np.unique(labels).size
        if cluster_count < 2 or cluster_count >= len(X):
            print(
                f"K={k}: silhouette score undefined "
                f"({cluster_count} distinct clusters); skipping"
            )
            scores.append({"K": k, "SilhouetteScore": np.nan})
            continue

        score = silhouette_score(X, labels)
        scores.append({"K": k, "SilhouetteScore": score})
        print(f"K={k}: Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    pd.DataFrame(scores).to_csv(reports_dir / "silhouette_scores.csv", index=False)
    if best_k is None:
        raise ValueError("No candidate K produced a valid silhouette score")

    print(f"\nHighest silhouette score among tested K values: K={best_k}")
    print(f"Fitting final K-Means model with K={best_k}...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X)
    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    print("\n--- CLUSTER SIZES ---")
    print(cluster_sizes.rename("PixelCount"))
    if cluster_sizes.min() == 1:
        print(
            "Warning: the selected solution contains a single-pixel cluster; "
            "interpret it cautiously as an exploratory result."
        )

    df_pca["Cluster"] = final_labels
    df_original["Cluster"] = final_labels

    clustered_path = processed_dir / "clustered_dataset.csv"
    df_pca.to_csv(clustered_path, index=False)
    print(f"\nCluster assignments saved to: {clustered_path}")

    cluster_profiles = df_original.groupby("Cluster")[RAW_FEATURES].mean()
    cluster_profiles.insert(
        0,
        "PixelCount",
        df_original.groupby("Cluster").size(),
    )
    profiles_path = reports_dir / "cluster_profiles.csv"
    cluster_profiles.to_csv(profiles_path)
    print("\n--- CLUSTER PROFILES (mean raw feature values) ---")
    print(cluster_profiles.round(2))
    print(f"Cluster profiles saved to: {profiles_path}")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=60,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(f"Exploratory Clusters in PC Space (K={best_k})", fontsize=14, pad=15)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster")
    ax.grid(True, linestyle="--", alpha=0.5)

    scatter_path = figures_dir / "cluster_scatter.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Cluster scatter plot saved to: {scatter_path}")
    print(f"Candidate silhouette scores saved to: {reports_dir / 'silhouette_scores.csv'}")


if __name__ == "__main__":
    perform_clustering()
