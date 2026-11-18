# Eco-AI | Blue Carbon Spatial Intelligence

An exploratory geospatial data-science project for inspecting mangrove raster
data in Mexico. The pipeline aligns selected raster layers, extracts complete
pixel observations, applies principal component analysis (PCA), and explores
K-Means groupings in feature and geographic space.

> **Live dashboard:** [Open Blue Carbon Spatial Intelligence](https://blue-carbon-monitoring-hmqkwkajrvfw7ebtgpdtn8.streamlit.app/)

## Project overview

The project turns aligned raster layers into an interactive dashboard for
reviewing feature distributions, PCA results, and spatial cluster assignments.
It is a prototype for data exploration; the current analysis does not establish
validated ecological classes or conservation priorities.

## Results and limitations

- The selected raster reference grid is **705 × 695 cells**, with a pixel size
  of **1,000 × 1,000 metres** in its projected coordinate reference system.
- **389 pixels** have valid values across all six selected layers. This is a
  small subset of the 489,975 reference-grid cells, so findings should not be
  generalized beyond the included observations without further investigation.
- `GreenTrend` is constant at 15 in these complete-pixel observations. Its
  correlation is undefined, and it is excluded from PCA and clustering.
- PCA is fit on five standardized features: `AGC`, `SOCD`, `NDVI`, `MTemp`, and
  `TotPrcp`. The first four components explain **88.50%** of variance in this
  dataset. This statistic alone does not establish that the remaining variance
  is ecologically unimportant.
- K-Means was evaluated for K=2 through K=8 using silhouette scores. K=3 scored
  **0.3233**, narrowly above K=2 (**0.3208**), but the selected K=3 solution
  contains a **single-pixel cluster** (cluster sizes: 240, 148, and 1). Treat
  this as an exploratory outlier, not a confirmed ecological zone.

Cluster labels and maps are exploratory model outputs. They have not been
independently validated against field observations, authoritative habitat
boundaries, or expert ecological labels.

## Dashboard

The Streamlit app includes:

- An executive overview with dataset and PCA metrics.
- Exploratory feature statistics, distributions, and correlations.
- PCA explained-variance and loading summaries.
- Interactive geographic and principal-component cluster views.
- Silhouette scores and cluster profiles, with the singleton cluster clearly
  identified.

The dashboard uses the processed data and reports committed to this repository;
it does not need the original archive to start.

## Run locally

Requirements: Python 3.9 or newer.

```bash
git clone https://github.com/rishi4842/blue-carbon-monitoring.git
cd blue-carbon-monitoring
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. Keep
the terminal running while you use the dashboard.

## Rebuild the data pipeline

The original dataset archive and extracted source rasters are intentionally
excluded from Git. To rebuild the processed outputs, obtain
`CStocks_Greenness_Mangroves_MX_1853.zip` from its source and place it in
`data/raw/`. Then, from the project root:

```bash
mkdir -p data/extracted
unzip -o data/raw/CStocks_Greenness_Mangroves_MX_1853.zip -d data/extracted/
```

Install the geospatial and analysis dependencies before running the pipeline:

```bash
python -m pip install rasterio numpy pandas scikit-learn matplotlib seaborn
```

Run the scripts in order:

```bash
python src/align_rasters.py
python src/create_dataset.py
python src/eda.py
python src/pca.py
python src/clustering.py
python src/mapping.py
```

`align_rasters.py` uses the first band of each selected input raster and aligns
it to the first-band NDVI reference grid using nearest-neighbour resampling.
Review the source data's band meanings and nodata metadata before interpreting
the results.

## Repository structure

```text
.
├── app.py
├── .streamlit/
│   └── config.toml
├── data/
│   └── processed/          # Aligned rasters and analysis tables used by the app
├── reports/                # PCA, EDA, clustering tables and figures
├── requirements.txt        # Streamlit dashboard dependencies
└── src/
    ├── align_rasters.py
    ├── create_dataset.py
    ├── eda.py
    ├── pca.py
    ├── clustering.py
    └── mapping.py
```

## Technology

- **Geospatial:** Rasterio, NumPy
- **Data analysis:** pandas, scikit-learn
- **Visualization and app:** Plotly, Streamlit, Matplotlib, Seaborn

