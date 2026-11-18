import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def align_all_rasters():
    base = "data/extracted/CStocks_Greenness_Mangroves_MX_1853/data"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # Map original NASA filenames to clean feature names.
    names_map = {
        "AGC_ARsw_2007.tif": "AGC.tif",
        "SOCD_ARsw_2010.tif": "SOCD.tif",
        "NDVI_ARsw_2001_2015.tif": "NDVI.tif",
        "GreenTrend_ARsw_2001_2015.tif": "GreenTrend.tif",
        "MTemp_ARsw_2001_2015.tif": "MTemp.tif",
        "TotPrcp_ARsw_2001_2015.tif": "TotPrcp.tif",
    }

    # Use NDVI as the reference grid for all output layers.
    ref_path = os.path.join(base, "NDVI_ARsw_2001_2015.tif")
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0)

        for in_name, out_name in names_map.items():
            in_path = os.path.join(base, in_name)
            out_path = os.path.join(out_dir, out_name)

            with rasterio.open(in_path) as src:
                # Initialize an empty array matching the reference dimensions.
                arr = np.full(
                    (ref.height, ref.width), -9999.0, dtype="float32"
                )

                # Reproject the first input band onto the reference grid.
                reproject(
                    source=src.read(1),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_nodata=-9999.0,
                    resampling=Resampling.nearest,
                )

                # Write the 2D array to the output's first band.
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr, 1)

            print(f"Aligned and saved: {out_name}")

    print("\nAll 6 layers aligned to the 705x695 NDVI reference grid.")


if __name__ == "__main__":
    align_all_rasters()
