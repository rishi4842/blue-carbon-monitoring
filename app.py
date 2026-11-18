from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_FEATURES = ["AGC", "SOCD", "NDVI", "MTemp", "TotPrcp"]


st.set_page_config(
    page_title="Blue Carbon Monitoring",
    page_icon="🌊",
    layout="wide",
)


@st.cache_data
def load_data():
    paths = {
        "observations": PROCESSED_DIR / "final_dataset.csv",
        "pca": PROCESSED_DIR / "pca_dataset.csv",
        "clusters": PROCESSED_DIR / "clustered_dataset.csv",
        "profiles": REPORTS_DIR / "cluster_profiles.csv",
        "variance": REPORTS_DIR / "pca_explained_variance.csv",
        "silhouette": REPORTS_DIR / "silhouette_scores.csv",
        "loadings": REPORTS_DIR / "pca_loadings.csv",
    }
    missing_files = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Required pipeline outputs are missing: " + ", ".join(missing_files)
        )

    data = {name: pd.read_csv(path) for name, path in paths.items()}
    required_columns = {
        "observations": {"Latitude", "Longitude", *RAW_FEATURES, "GreenTrend"},
        "pca": {"Latitude", "Longitude", "PC1", "PC2", "PC3", "PC4", "PC5"},
        "clusters": {"Latitude", "Longitude", "PC1", "PC2", "Cluster"},
    }
    for name, columns in required_columns.items():
        missing_columns = columns.difference(data[name].columns)
        if missing_columns:
            raise ValueError(
                f"{name} data is missing columns: {', '.join(sorted(missing_columns))}"
            )

    return data


try:
    data = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Dashboard data could not be loaded: {error}")
    st.info(
        "Run the data pipeline from the project root: create_dataset.py, "
        "eda.py, pca.py, then clustering.py."
    )
    st.stop()

observations = data["observations"]
pca_data = data["pca"]
cluster_data = data["clusters"].copy()
profiles = data["profiles"]
variance = data["variance"]
silhouette = data["silhouette"]
loadings = data["loadings"]

if not (
    len(observations) == len(pca_data) == len(cluster_data)
    and observations[["Latitude", "Longitude"]].equals(
        pca_data[["Latitude", "Longitude"]]
    )
    and observations[["Latitude", "Longitude"]].equals(
        cluster_data[["Latitude", "Longitude"]]
    )
):
    st.error("Pipeline CSVs do not refer to the same pixels in the same row order.")
    st.stop()

cluster_data["Cluster"] = cluster_data["Cluster"].astype(str)
cluster_sizes = cluster_data["Cluster"].value_counts().sort_index()
singleton_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
best_silhouette_row = silhouette.loc[silhouette["SilhouetteScore"].idxmax()]
best_k = int(best_silhouette_row["K"])
four_pc_variance = float(
    variance.loc[variance["Component"].isin(["PC1", "PC2", "PC3", "PC4"]),
                 "CumulativeVariancePercent"].max()
)

st.sidebar.title("🌊 Blue Carbon Monitoring")
st.sidebar.caption("PCA and spatial clustering project")
page = st.sidebar.radio(
    "Explore",
    ["Overview", "Data exploration", "PCA", "Clustering and map"],
)

st.title("Mangrove Raster Analysis")
st.caption(
    "Interactive exploration of processed raster pixels. Cluster labels are "
    "algorithmic groups, not independently validated ecological classifications."
)

if page == "Overview":
    st.header("Project overview")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Pixels with complete data", f"{len(observations):,}")
    metric_2.metric("Exploratory clusters", f"{cluster_sizes.size}")
    metric_3.metric("Variance in first 4 PCs", f"{four_pc_variance:.2f}%")

    st.markdown(
        """
        **Workflow**
        1. Align selected raster layers to a common NDVI grid.
        2. Flatten cells with valid values across all selected features.
        3. Inspect feature distributions and correlations.
        4. Standardize variable features and calculate principal components.
        5. Compare K-Means cluster counts and inspect their spatial distribution.
        """
    )
    st.warning(
        "The current table contains 389 complete pixels. `GreenTrend` is "
        "constant in this sample and is excluded from PCA. The K=3 result "
        "includes a single-pixel cluster; treat all clusters as exploratory."
    )
    st.subheader("Sample observations")
    st.dataframe(observations.head(20), width="stretch")

elif page == "Data exploration":
    st.header("Data exploration")
    st.write("Summary statistics for the five variable features used in PCA.")
    st.dataframe(
        observations[RAW_FEATURES].describe().T,
        width="stretch",
    )
    st.subheader("Feature distributions")
    distributions_path = FIGURES_DIR / "feature_distributions.png"
    if distributions_path.is_file():
        st.image(str(distributions_path), width="stretch")
    st.subheader("Feature correlations")
    correlation_path = FIGURES_DIR / "correlation_heatmap.png"
    if correlation_path.is_file():
        st.image(str(correlation_path), width="stretch")
    st.info(
        "`GreenTrend` is constant at 15 in the current complete-pixel dataset. "
        "Its correlation is undefined, and it is not included in PCA."
    )

elif page == "PCA":
    st.header("Principal component analysis")
    st.write(
        "The five variable features were standardized before PCA. The table "
        "below reports the variance explained by each component."
    )
    variance_display = variance.copy()
    variance_display["Component"] = variance_display["Component"].astype(str)
    st.dataframe(variance_display, width="stretch", hide_index=True)

    scree_path = FIGURES_DIR / "pca_variance.png"
    if scree_path.is_file():
        st.image(str(scree_path), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Feature loadings")
        st.dataframe(loadings, width="stretch")
    with right:
        st.subheader("Transformed pixel values")
        st.dataframe(
            pca_data[["Latitude", "Longitude", "PC1", "PC2", "PC3", "PC4"]].head(20),
            width="stretch",
        )
    st.caption(
        f"The first four components account for {four_pc_variance:.2f}% "
        "of variance in this dataset; this does not establish that the "
        "discarded variance is ecologically unimportant."
    )

elif page == "Clustering and map":
    st.header("Exploratory clustering")
    st.write(
        f"Among the tested K values, K={best_k} had the highest silhouette "
        f"score ({best_silhouette_row['SilhouetteScore']:.4f}). This is a "
        "model-selection comparison, not scientific validation of ecological zones."
    )
    st.dataframe(profiles, width="stretch")
    st.subheader("Silhouette scores by candidate K")
    st.line_chart(silhouette.set_index("K")["SilhouetteScore"])

    if singleton_clusters:
        st.warning(
            "Singleton cluster(s): "
            + ", ".join(singleton_clusters)
            + ". They are shown separately below; their ecological meaning "
            "has not been established."
        )

    available_clusters = sorted(cluster_data["Cluster"].unique())
    selected_clusters = st.multiselect(
        "Clusters to display",
        options=available_clusters,
        default=available_clusters,
    )
    visible = cluster_data[cluster_data["Cluster"].isin(selected_clusters)]
    st.subheader("Interactive spatial view")
    st.caption(
        "Longitude and latitude are plotted as coordinates, not on a basemap. "
        "The chart supports zooming and panning."
    )
    spatial_chart = (
        alt.Chart(visible)
        .mark_circle(size=85, opacity=0.8)
        .encode(
            x=alt.X("Longitude:Q", scale=alt.Scale(zero=False), title="Longitude"),
            y=alt.Y("Latitude:Q", scale=alt.Scale(zero=False), title="Latitude"),
            color=alt.Color("Cluster:N", title="Cluster"),
            tooltip=[
                alt.Tooltip("Cluster:N"),
                alt.Tooltip("Latitude:Q", format=".5f"),
                alt.Tooltip("Longitude:Q", format=".5f"),
                *[
                    alt.Tooltip(f"{feature}:Q", format=".3f")
                    for feature in RAW_FEATURES
                ],
            ],
        )
        .properties(height=500)
        .interactive()
    )
    st.altair_chart(spatial_chart, use_container_width=True)

    left, right = st.columns(2)
    scatter_path = FIGURES_DIR / "cluster_scatter.png"
    map_path = FIGURES_DIR / "spatial_cluster_map.png"
    with left:
        st.subheader("Clusters in principal-component space")
        if scatter_path.is_file():
            st.image(str(scatter_path), width="stretch")
    with right:
        st.subheader("Static spatial cluster map")
        if map_path.is_file():
            st.image(str(map_path), width="stretch")
