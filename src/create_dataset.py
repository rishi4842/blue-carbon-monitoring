from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


FEATURES = ["AGC", "SOCD", "NDVI", "GreenTrend", "MTemp", "TotPrcp"]


def create_ml_dataset():
    project_dir = Path(__file__).resolve().parents[1]
    processed_dir = project_dir / "data" / "processed"

    print("Reading reference grid to calculate geographic coordinates...")
    with rasterio.open(processed_dir / "NDVI.tif") as ref:
        width = ref.width
        height = ref.height
        ref_crs = ref.crs
        ref_transform = ref.transform
        pixel_count = width * height
        rows, cols = np.indices((height, width), dtype=np.float64)

        # Affine coordinates use pixel corners; add 0.5 to get pixel centers.
        pixel_cols = cols + 0.5
        pixel_rows = rows + 0.5
        xs = (
            ref_transform.c
            + pixel_cols * ref_transform.a
            + pixel_rows * ref_transform.b
        ).ravel()
        ys = (
            ref_transform.f
            + pixel_cols * ref_transform.d
            + pixel_rows * ref_transform.e
        ).ravel()

        print("Transforming projected coordinates to Latitude/Longitude...")
        longitudes, latitudes = transform(
            ref_crs, "EPSG:4326", xs.tolist(), ys.tolist()
        )

    valid_mask = np.ones(pixel_count, dtype=bool)
    data = {
        "Latitude": np.asarray(latitudes),
        "Longitude": np.asarray(longitudes),
    }

    for feature in FEATURES:
        print(f"Flattening {feature}.tif...")
        raster_path = processed_dir / f"{feature}.tif"
        with rasterio.open(raster_path) as src:
            if (
                src.width != width
                or src.height != height
                or src.crs != ref_crs
                or src.transform != ref_transform
            ):
                raise ValueError(
                    f"{raster_path} does not match the NDVI reference grid"
                )
            if src.count < 1:
                raise ValueError(f"{raster_path} contains no raster bands")

            masked = src.read(1, masked=True)
            valid_mask &= ~np.ma.getmaskarray(masked).ravel()
            values = np.ma.asarray(masked, dtype=np.float32).filled(np.nan)
            values = values.ravel()
            valid_mask &= np.isfinite(values)
            data[feature] = values

    print("Building DataFrame and cleaning invalid pixels...")
    df = pd.DataFrame(data)
    df_clean = df.loc[valid_mask].dropna().copy()

    total_pixels = pixel_count
    valid_pixels = len(df_clean)
    output_path = processed_dir / "final_dataset.csv"
    df_clean.to_csv(output_path, index=False)

    print("\n--- DATASET SUMMARY ---")
    print(f"Total theoretical pixels ({width}x{height}): {total_pixels:,}")
    print(f"Valid pixels across all features:           {valid_pixels:,}")
    print(f"Compression ratio (valid/total):             {valid_pixels / total_pixels * 100:.2f}%")
    print(f"\nML dataset saved to: {output_path}")
    print("\nFirst 3 rows:")
    print(df_clean.head(3))


if __name__ == "__main__":
    create_ml_dataset()
