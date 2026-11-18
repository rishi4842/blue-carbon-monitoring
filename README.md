# Blue Carbon Monitoring Dashboard

Streamlit dashboard for exploring processed mangrove raster observations, PCA
results, and exploratory K-Means cluster assignments.

## Run locally

From the project root:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.
Keep the terminal running while using the dashboard.

## Deploy with Streamlit Community Cloud

The app reads the processed CSVs and report figures committed in `data/processed/`
and `reports/`. The large original archive, extracted source rasters, and local
virtual environment are excluded by `.gitignore`.

1. Create an empty GitHub repository.
2. Initialize Git in this project, commit the project files, and push to that
   repository. Review the staged files before committing to confirm
   `data/raw/`, `data/extracted/`, and `.venv/` are excluded.
3. In Streamlit Community Cloud, select that GitHub repository, branch, and
   `app.py` as the app file, then deploy.

The committed processed outputs are small enough for this dashboard and allow it
to start without rerunning the raster-processing pipeline.

## Data pipeline

The scripts under `src/` produce the processed tables and visualizations consumed
by the dashboard. Run them in order when rebuilding the data:

```bash
.venv/bin/python src/align_rasters.py
.venv/bin/python src/create_dataset.py
.venv/bin/python src/eda.py
.venv/bin/python src/pca.py
.venv/bin/python src/clustering.py
.venv/bin/python src/mapping.py
```

The clustering and mapped groups are exploratory results, not independently
validated ecological classifications.