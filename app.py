from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_FEATURES = ["AGC", "SOCD", "NDVI", "GreenTrend", "MTemp", "TotPrcp"]
PCA_FEATURES = ["PC1", "PC2", "PC3", "PC4", "PC5"]
COORDINATES = ["Latitude", "Longitude"]


st.set_page_config(
    page_title="Blue Carbon | Spatial Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(14, 116, 144, .17), transparent 38%),
            #071521;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1c2a 0%, #091723 100%);
        border-right: 1px solid rgba(148, 193, 214, .13);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(16, 37, 54, .96), rgba(11, 29, 43, .96));
        border: 1px solid rgba(125, 190, 218, .17);
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] p {
        color: #a9c5d5;
        font-size: .82rem;
        letter-spacing: .04em;
    }
    [data-testid="stMetricValue"] {
        color: #f0f8fc;
    }
    .hero {
        padding: 26px 30px;
        margin: 4px 0 22px 0;
        border: 1px solid rgba(56, 189, 248, .25);
        border-radius: 18px;
        background: linear-gradient(115deg, rgba(14, 116, 144, .23), rgba(16, 37, 54, .64));
    }
    .eyebrow {
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .72rem;
        font-weight: 700;
    }
    .hero h1 {
        margin: 8px 0 8px 0;
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -.04em;
    }
    .hero p {
        color: #bad1de;
        font-size: 1.02rem;
        max-width: 760px;
        margin-bottom: 0;
    }
    .section-note {
        color: #a9c5d5;
        font-size: .92rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(125, 190, 218, .16);
        border-radius: 12px;
        background: rgba(16, 37, 54, .36);
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    missing_files = [
        str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required pipeline outputs are missing: " + ", ".join(missing_files)
        )

    data = {name: pd.read_csv(path) for name, path in paths.items()}
    required_columns = {
        "observations": {*COORDINATES, *RAW_FEATURES},
        "pca": {*COORDINATES, *PCA_FEATURES},
        "clusters": {*COORDINATES, *PCA_FEATURES, "Cluster"},
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
        "Run the data pipeline to generate the processed tables and reports, "
        "then reload the dashboard."
    )
    st.stop()

observations = data["observations"]
pca_data = data["pca"]
cluster_data = data["clusters"].copy()
profiles = data["profiles"]
variance = data["variance"]
silhouette = data["silhouette"]
loadings = data["loadings"]

expected_coordinates = observations[COORDINATES]
expected_index = pd.MultiIndex.from_frame(expected_coordinates.round(6))
if expected_index.has_duplicates:
    st.error("Observation coordinates are not unique at six-decimal precision.")
    st.stop()

for name, frame in [("PCA", pca_data), ("cluster", cluster_data)]:
    if len(frame) != len(observations):
        st.error(
            f"{name} data contains {len(frame)} rows, but observations contain "
            f"{len(observations)}."
        )
        st.stop()

    coordinate_index = pd.MultiIndex.from_frame(frame[COORDINATES].round(6))
    if coordinate_index.has_duplicates or set(coordinate_index) != set(expected_index):
        st.error(
            f"{name} data does not contain exactly the same geographic pixels "
            "as the observation dataset."
        )
        st.stop()

    frame.index = coordinate_index
    frame = frame.reindex(expected_index).reset_index(drop=True)
    frame[COORDINATES] = expected_coordinates.to_numpy()
    if name == "PCA":
        pca_data = frame
    else:
        cluster_data = frame

cluster_data["Cluster"] = cluster_data["Cluster"].astype(str)
cluster_sizes = cluster_data["Cluster"].value_counts().sort_index()
singleton_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
variable_features = [
    feature for feature in RAW_FEATURES if observations[feature].nunique() > 1
]
constant_features = [
    feature for feature in RAW_FEATURES if observations[feature].nunique() <= 1
]
best_silhouette_row = silhouette.loc[silhouette["SilhouetteScore"].idxmax()]
best_k = int(best_silhouette_row["K"])
four_pc_variance = float(
    variance.loc[
        variance["Component"].isin(["PC1", "PC2", "PC3", "PC4"]),
        "CumulativeVariancePercent",
    ].max()
)


def show_figure(filename: str, caption: Optional[str] = None) -> None:
    path = FIGURES_DIR / filename
    if path.is_file():
        st.image(str(path), width="stretch", caption=caption)
    else:
        st.warning(f"Figure not available: {filename}")


with st.sidebar:
    st.markdown("## 🌊 Blue Carbon")
    st.caption("SPATIAL INTELLIGENCE · SIH26024")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        [
            "Executive Summary",
            "Data Intelligence",
            "Dimensionality Reduction",
            "Geospatial Clustering",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("MANGROVE RASTER ANALYSIS")
    st.caption("PCA · K-Means · Spatial exploration")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Coastal ecosystem analytics · Mexico</div>
      <h1>Blue Carbon Intelligence</h1>
      <p>From aligned environmental rasters to interpretable feature reduction
      and exploratory spatial clusters.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Executive Summary":
    st.subheader("Pipeline at a glance")
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Complete pixels", f"{len(observations):,}")
    kpi_2.metric("Variable features", f"{len(variable_features)}")
    kpi_3.metric("Grid cell size", "1 km × 1 km")
    kpi_4.metric("Variance in 4 PCs", f"{four_pc_variance:.2f}%")

    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("A transparent analytical workflow")
        st.markdown(
            """
            **01 · Align** — six raster layers are projected to a shared 705 × 695
            reference grid.

            **02 · Inspect** — feature distributions and correlations are reviewed
            before modeling.

            **03 · Reduce** — variable features are standardized and transformed
            into principal components.

            **04 · Explore** — K-Means assignments are compared and mapped back
            to geographic coordinates.
            """
        )
    with right:
        st.subheader("Model signals")
        st.metric("Highest tested silhouette", f"{best_silhouette_row['SilhouetteScore']:.3f}", f"K = {best_k}")
        st.caption(
            "A relative score for tested clusterings—not independent validation "
            "of ecological zones."
        )

    if constant_features:
        st.warning(
            "Constant feature(s) are excluded from PCA: "
            + ", ".join(constant_features)
            + ". Confirm their source data before drawing ecological conclusions."
        )
    if singleton_clusters:
        st.info(
            "The selected clustering includes singleton cluster(s): "
            + ", ".join(singleton_clusters)
            + ". These are shown as exploratory outliers, not confirmed zones."
        )

    with st.expander("Inspect processed observations"):
        st.dataframe(observations, width="stretch", hide_index=True)

elif page == "Data Intelligence":
    st.subheader("Data intelligence")
    st.markdown(
        '<p class="section-note">Review the distributions, relationships, and '
        "feature quality before interpreting model outputs.</p>",
        unsafe_allow_html=True,
    )
    tab_correlations, tab_distributions, tab_records = st.tabs(
        ["Correlations", "Distributions", "Feature statistics"]
    )
    with tab_correlations:
        show_figure("correlation_heatmap.png")
        if constant_features:
            st.info(
                "Correlation is undefined for constant feature(s): "
                + ", ".join(constant_features)
                + ". These features do not enter PCA."
            )
    with tab_distributions:
        show_figure("feature_distributions.png")
    with tab_records:
        st.dataframe(
            observations[RAW_FEATURES].describe().T,
            width="stretch",
        )
        with st.expander("Preview source rows"):
            st.dataframe(observations.head(50), width="stretch", hide_index=True)

elif page == "Dimensionality Reduction":
    st.subheader("Principal component analysis")
    st.markdown(
        '<p class="section-note">Standardized variable features projected into '
        "orthogonal components.</p>",
        unsafe_allow_html=True,
    )
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Input features", f"{len(variable_features)}")
    metric_2.metric("Components retained", f"{len(PCA_FEATURES)}")
    metric_3.metric("Variance in first 4 PCs", f"{four_pc_variance:.2f}%")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Explained variance")
        show_figure("pca_variance.png")
    with right:
        st.subheader("Component summary")
        st.dataframe(variance, width="stretch", hide_index=True)

    with st.expander("Inspect feature loadings"):
        st.caption(
            "Loadings describe each feature's contribution to a component; "
            "their signs can flip without changing the PCA solution."
        )
        st.dataframe(loadings, width="stretch")
    with st.expander("Preview transformed pixel values"):
        st.dataframe(
            pca_data[COORDINATES + PCA_FEATURES].head(50),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Variance retained is a statistical summary; it does not establish that "
        "the omitted variance is ecologically unimportant."
    )

elif page == "Geospatial Clustering":
    st.subheader("Geospatial clustering")
    st.markdown(
        '<p class="section-note">Explore K-Means assignments in geographic and '
        "principal-component space.</p>",
        unsafe_allow_html=True,
    )
    st.metric(
        "Highest silhouette among tested K",
        f"{best_silhouette_row['SilhouetteScore']:.3f}",
        f"K = {best_k}",
    )
    if singleton_clusters:
        st.warning(
            "Singleton cluster(s): "
            + ", ".join(singleton_clusters)
            + ". A silhouette-score advantage does not establish that a "
            "single-pixel group is an ecological zone."
        )

    tab_map, tab_profiles = st.tabs(["Spatial view", "Cluster diagnostics"])
    with tab_map:
        available_clusters = sorted(cluster_data["Cluster"].unique())
        selected_clusters = st.multiselect(
            "Clusters to show",
            options=available_clusters,
            default=available_clusters,
        )
        visible = cluster_data[cluster_data["Cluster"].isin(selected_clusters)]
        if not visible.empty:
            st.map(
                visible[COORDINATES],
                latitude="Latitude",
                longitude="Longitude",
                height=480,
            )
        else:
            st.info("Select one or more clusters to display their pixels.")
        st.caption(
            f"Showing {len(visible):,} of {len(cluster_data):,} pixels. "
            "The cluster-colored static map is shown below."
        )
        show_figure("spatial_cluster_map.png")
    with tab_profiles:
        st.subheader("Mean feature values by cluster")
        st.dataframe(profiles, width="stretch")
        st.subheader("Silhouette scores by candidate K")
        st.line_chart(silhouette.set_index("K")["SilhouetteScore"])
        st.subheader("Clusters in principal-component space")
        show_figure("cluster_scatter.png")
