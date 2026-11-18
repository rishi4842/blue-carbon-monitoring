from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_spatial_map():
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "data" / "processed" / "clustered_dataset.csv"
    figures_dir = project_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clustered data from {data_path}...")
    df = pd.read_csv(data_path)

    required_columns = ["Latitude", "Longitude", "Cluster"]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Clustered dataset is missing columns: {missing_columns}")
    if df.empty:
        raise ValueError("Clustered dataset contains no rows to map")

    coordinates = df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Latitude/longitude coordinates contain non-finite values")
    if not df["Latitude"].between(-90, 90).all():
        raise ValueError("Latitude values must be between -90 and 90 degrees")
    if not df["Longitude"].between(-180, 180).all():
        raise ValueError("Longitude values must be between -180 and 180 degrees")

    cluster_sizes = df["Cluster"].value_counts().sort_index()
    singleton_clusters = set(cluster_sizes[cluster_sizes == 1].index)
    palette = sns.color_palette("Set1", n_colors=len(cluster_sizes))
    colors = dict(zip(cluster_sizes.index, palette))

    fig, ax = plt.subplots(figsize=(10, 8))
    for cluster, size in cluster_sizes.items():
        points = df.loc[df["Cluster"] == cluster]
        if cluster in singleton_clusters:
            label = f"Cluster {cluster} — singleton outlier (1 pixel)"
            marker = "*"
            marker_size = 240
            zorder = 3
        else:
            label = f"Cluster {cluster} (n={size})"
            marker = "o"
            marker_size = 60
            zorder = 2

        ax.scatter(
            points["Longitude"],
            points["Latitude"],
            color=colors[cluster],
            label=label,
            marker=marker,
            s=marker_size,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=zorder,
        )

    mean_latitude = float(df["Latitude"].mean())
    latitude_cosine = np.cos(np.deg2rad(mean_latitude))
    if latitude_cosine <= 0:
        raise ValueError("Cannot create a local lon/lat map at this latitude")

    ax.set_title("Spatial Distribution of Exploratory K-Means Clusters", pad=15)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    # Compensate for longitude degrees being shorter at the mean latitude.
    ax.set_aspect(1 / latitude_cosine, adjustable="datalim")
    ax.legend(title="Cluster (pixel count)", loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = figures_dir / "spatial_cluster_map.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n--- MAPPED CLUSTER COUNTS ---")
    print(cluster_sizes.rename("PixelCount"))
    if singleton_clusters:
        print(
            "Singleton clusters are highlighted as outliers; "
            "cluster membership alone does not establish an ecological zone."
        )
    print(f"\nSpatial map saved to: {output_path}")


if __name__ == "__main__":
    create_spatial_map()
