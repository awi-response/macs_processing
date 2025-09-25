import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from osgeo import gdal
from shapely.geometry import shape


def convert_to_cog(
    infile: str | Path,
    outfile: str | Path,
    bigtiff: bool = True,
    compress: str = "DEFLATE",
    num_threads: str = "ALL_CPUS",
):
    """
    Convert a GeoTIFF file to Cloud Optimized GeoTIFF (COG) format using GDAL Python bindings.

    Parameters:
    input_file (str): Path to input GeoTIFF file
    output_file (str): Path to output COG file
    bigtiff (bool): Whether to use BIGTIFF=YES
    compress (str): Compression method (default: 'DEFLATE')
    num_threads (str): Number of threads to use (default: 'ALL_CPUS')

    Returns:
    bool: True if successful, False otherwise
    """

    try:
        # Set GDAL configuration options
        gdal.SetConfigOption("BIGTIFF", "YES" if bigtiff else "NO")
        gdal.SetConfigOption("COMPRESS", compress)
        gdal.SetConfigOption("NUM_THREADS", num_threads)

        # Open input dataset
        src_ds = gdal.Open(infile)
        if src_ds is None:
            print(f"Could not open input file: {infile}")
            return False

        # Create COG driver
        driver = gdal.GetDriverByName("COG")
        if driver is None:
            print("COG driver not available")
            return False

        # Set creation options
        creation_options = [
            f"COMPRESS={compress}",
            f"NUM_THREADS={num_threads}",
            "BIGTIFF=YES" if bigtiff else "BIGTIFF=NO",
        ]

        # Create output dataset
        dst_ds = driver.CreateCopy(outfile, src_ds, options=creation_options)

        # Close datasets
        dst_ds = None
        src_ds = None

        print(f"Successfully converted {infile} to {outfile}")
        return True

    except Exception as e:
        print(f"Error converting file: {str(e)}")
        return False


def create_mask_vector_rasterio(
    raster_file: Path | str,
    temporary_target_dir: Path | str,
    remove_raster_mask: bool = False,
):
    """
    Function to create vector polygons for valid data and nodata/zero-value areas from a raster file using rasterio.

    This function generates a binary mask from the input raster, where valid data pixels (non-zero) are set to 1 and nodata/invalid pixels (zero) are set to 0.
    It then writes this mask to a temporary raster file, polygonizes the mask to extract vector polygons for both valid and invalid regions,
    and saves the resulting polygons as a GeoJSON file with a 'DN' attribute (255 for valid data, 0 for nodata/invalid).
    Optionally, the temporary mask raster can be deleted after processing.

    Parameters
    ----------
    raster_file : Path or str
        Input raster file from which to create the vector mask.
    temporary_target_dir : Path or str
        Directory path where temporary files should be stored.
    remove_raster_mask : bool, optional
        If True, delete the temporary raster mask after vectorization. Default is False.

    Returns
    -------
    mask_vector : Path
        Path to the output GeoJSON file containing the vectorized mask polygons.
    """
    raster_file = Path(raster_file)
    temporary_target_dir = Path(temporary_target_dir)

    maskfile = temporary_target_dir / (raster_file.stem + "_mask.tif")
    mask_vector = temporary_target_dir / (raster_file.stem + "_mask.geojson")

    try:
        # Read original raster and create mask
        with rasterio.open(raster_file, 'r') as src:
            # Read first band
            raster_data = src.read(1)

            # Create binary mask: 1 for valid data, 0 for nodata/invalid
            binary_mask = (raster_data != 0).astype(np.uint8)

            # Write mask to temporary file
            profile = src.profile.copy()
            profile.update({"dtype": "uint8", "count": 1, "nodata": 0, "driver": "GTiff"})

            with rasterio.open(maskfile, "w", options={'IGNORE_COG_LAYOUT_BREAK': 'YES'}, **profile) as dst:
                dst.write(binary_mask, 1)

        # Polygonize using rasterio features - create polygons for both valid and invalid areas
        with rasterio.open(maskfile) as src:
            mask_data = src.read(1)
            transform = src.transform

            # Extract shapes (polygons) from raster
            shapes = rasterio.features.shapes(
                mask_data,
                mask=mask_data >= 0,  # Include all values (0 and 1)
                transform=transform,
            )

            # Convert to GeoDataFrame with DN attribute
            geoms = []
            dn_values = []

            for shape_dict, value in shapes:
                # Convert GeoJSON dict to Shapely geometry
                geom = shape(shape_dict)
                geoms.append(geom)
                # Set DN value: 255 for valid data (value = 1), 0 for nodata/invalid (value = 0)
                dn_values.append(255 if value > 0 else 0)

            if geoms:  # Only create GeoDataFrame if we have geometries
                # Create GeoDataFrame with proper structure including DN attribute
                gdf = gpd.GeoDataFrame(
                    {"geometry": geoms, "DN": dn_values}, crs=src.crs
                )
                gdf.to_file(mask_vector, driver="GeoJSON")
            else:
                # Create empty GeoJSON if no polygons found
                empty_gdf = gpd.GeoDataFrame({"geometry": [], "DN": []}, crs=src.crs)
                empty_gdf.to_file(mask_vector, driver="GeoJSON")

        # Remove temporary files if requested
        if remove_raster_mask and maskfile.exists():
            maskfile.unlink()

    except Exception as e:
        # Clean up temporary files on error
        try:
            if maskfile.exists():
                maskfile.unlink()
        except:
            pass
        raise e

    return mask_vector


# deprecated function which we want to avoid
def create_mask_vector(
    raster_file,
    temporary_target_dir,
    remove_raster_mask=False,
    polygonize=Path(os.environ["CONDA_PREFIX"]) / "Scripts" / "gdal_polygonize.py",
):
    """
    Function to create vectors of valid data for a raster file
    Parameters
    ----------
    raster_file : Path
        input raster file from which to create vector mask.
    temporary_target_dir : Path
        directory path where temporary files should be stored.
    remove_raster_mask : bool, optional
        delete temporary raster mask. The default is False.
    polygonize : Path, optional
        Path to gdal polygonize function, Automatically created for (Windows) conda environments. The default is Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'.

    Returns
    -------
    mask_vector : Path
        path of output vector footprints file.

    """
    maskfile = temporary_target_dir / (raster_file.stem + "_mask.tif")
    mask_vector = temporary_target_dir / (raster_file.stem + "_mask.geojson")

    s_extract_mask = (
        f"gdal_translate -q -ot Byte -b mask -of GTiff {raster_file} {maskfile}"
    )
    os.system(s_extract_mask)

    s_polygonize_mask = f"python {polygonize} -q -f GeoJSON {maskfile} {mask_vector}"
    os.system(s_polygonize_mask)

    # needs to be fixed - is not deleting at the moment
    if remove_raster_mask:
        os.remove(maskfile)

    return mask_vector
