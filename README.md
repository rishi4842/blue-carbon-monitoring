# PCA-Based Feature Compression of Coastal Ecosystem Sensor Data for Blue Carbon Zone Monitoring

[**Open the live interactive dashboard**](https://blue-carbon-monitoring-hmqkwkajrvfw7ebtgpdtn8.streamlit.app/)

## Project overview

This project explores how aligned environmental raster layers can be analyzed
together to study mangrove ecosystems in Mexico. A Python pipeline aligns
selected rasters, extracts pixels with complete observations, uses principal
component analysis (PCA) to summarize variable features, and applies K-Means
clustering to explore patterns in feature and geographic space.

The Streamlit dashboard presents the results for interactive inspection. The
current clusters are exploratory outputs, not validated ecological classes,
habitat boundaries, or conservation recommendations.

## Technical workflow

### 1. Geospatial data preparation

The `rasterio` pipeline aligns six selected GeoTIFF layers—above-ground carbon
(`AGC`), soil organic carbon density (`SOCD`), NDVI, greenness trend
(`GreenTrend`), mean temperature (`MTemp`), and total precipitation
(`TotPrcp`)—to the NDVI reference grid. The reference is **705 × 695 cells**
with **1,000 m × 1,000 m** pixels in its projected coordinate reference system.
Nearest-neighbour resampling is used, and pixels without valid values across
all six layers are excluded from the tabular dataset.

### 2. Exploratory data analysis

The pipeline summarizes feature distributions and correlations before PCA.
`GreenTrend` is constant at 15 in the current complete-pixel sample, so its
correlations are undefined and it is excluded from PCA and clustering.

### 3. Principal component analysis

The five variable features (`AGC`, `SOCD`, `NDVI`, `MTemp`, and `TotPrcp`) are
standardized with `StandardScaler` before PCA. The first four principal
components explain **88.50%** of the variance in the current dataset. Variance
retained is a statistical summary; it does not establish that the remaining
variance is ecologically unimportant.

### 4. Exploratory K-Means clustering

K-Means is evaluated for K=2 through K=8 using silhouette scores and the first
four principal components. K=3 has the highest tested score (**0.3233**), only
slightly higher than K=2 (**0.3208**). The K=3 result has cluster sizes of 240,
148, and **1 pixel**. The singleton is treated as an exploratory outlier; the
metric does not validate it as an ecological zone.

### 5. Interactive dashboard

The Streamlit dashboard includes interactive Plotly views for feature
distributions and correlations, PCA variance, cluster scores, and cluster
locations. It also provides cluster profiles and model diagnostics.

## Results and limitations

- The reference grid contains 489,975 cells; **389 pixels** have valid values
  across all six selected layers. This small complete-pixel sample limits how
  broadly results should be generalized.
- The source feature layers, their band semantics, and nodata metadata should
  be checked before interpreting values ecologically.
- Cluster assignments have not been independently validated against field
  observations, authoritative habitat maps, or expert ecological labels.
- The project is an exploratory analysis prototype, not a production
  monitoring or decision-support system.

## Technology

- **Geospatial processing:** Rasterio, NumPy
- **Data analysis and machine learning:** pandas, scikit-learn
- **Dashboard and visualization:** Streamlit, Plotly, Matplotlib, Seaborn
- **Core language:** Python 3.9+

## Run the dashboard locally

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

The source archive and extracted rasters are excluded from Git. Obtain
`CStocks_Greenness_Mangroves_MX_1853.zip` from its source and place it in
`data/raw/`. Extract it and install the additional geospatial processing
dependencies:

```bash
mkdir -p data/extracted
unzip -o data/raw/CStocks_Greenness_Mangroves_MX_1853.zip -d data/extracted/
python -m pip install rasterio numpy scikit-learn matplotlib seaborn
```

From the project root, run:

```bash
python src/align_rasters.py
python src/create_dataset.py
python src/eda.py
python src/pca.py
python src/clustering.py
python src/mapping.py
```

`align_rasters.py` uses the first band of each input raster and aligns it to the
first band of the NDVI reference grid. Review the source raster metadata and
band meanings before interpreting the output.

## Repository structure

```text
.
├── app.py
├── .streamlit/
│   └── config.toml
├── data/
│   └── processed/          # Aligned rasters and analysis tables used by app
├── reports/                # EDA, PCA, clustering results and figures
├── requirements.txt        # Dashboard dependencies
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
