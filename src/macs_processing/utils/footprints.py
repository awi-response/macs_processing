# import glob2, glob
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_kml(file, sensor_name="RGB", layer=0, looking="center"):
    df = gpd.read_file(file, driver="KML", layer=layer)
    df["Sensor"] = sensor_name
    df["Looking"] = looking
    df = df.set_index("Name", drop=True)
    return df


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dataset_footprints(
    dataset_path: Path, 
    CRS: str = "EPSG:4326", 
    regex_nav: str = "*nav.txt", 
    verbosity: int = 0
):
    """
    Processes KML files and generates GeoPackage outputs.

    Parameters:
    - dataset_path (Path): The path to the dataset folder containing KML and navigation files.
    - CRS (str): The Coordinate Reference System to be used for GeoDataFrames. Default is 'EPSG:4326'.
    - regex_nav (str): The regex pattern for navigation files. Default is '*nav.txt'.
    - verbosity (int): Level of verbosity for logging. 0 for quiet, 1 for project name only, 2 for full details.

    Returns:
    - None
    """

    dataset_id = dataset_path.name

    if verbosity >= 1:
        logging.info(f"Running footprint creation for project: {dataset_id}")

    # Load KML files
    file_kml = list(dataset_path.glob("*.kml"))
    file_nav = list(dataset_path.glob(regex_nav))

    # Check if files were found
    if not file_kml:
        logging.warning(f"No KML files found in {dataset_path}.")
        return  # Exit the function if no KML files are found

    if not file_nav:
        logging.warning(
            f"No navigation files found in {dataset_path} matching pattern '{regex_nav}'."
        )
        return  # Exit the function if no navigation files are found

    # Log the found files if verbosity is high enough
    if verbosity >= 2:
        logging.info(f"KML file found: {file_kml[0]}")
        logging.info(f"Navigation file found: {file_nav[0]}")

    # Load and concatenate data from KML files
    df_concat = gpd.GeoDataFrame(
        pd.concat(
            [
                load_kml(file_kml[0], sensor_name="RGB", layer=1, looking="center"),
                load_kml(file_kml[0], sensor_name="NIR", layer=0, looking="center"),
                load_kml(file_kml[0], sensor_name="TIR", layer=2, looking="center"),
            ]
        ),
        crs=CRS,
    )

    # Join navigation data with concatenated KML data
    df_nav = pd.read_csv(file_nav[0], sep="\s+")
    
    # Remove trailing whitespaces from column names
    df_nav.columns = df_nav.columns.str.strip()

    # Get filename and set as index for join
    df_nav["Name"] = df_nav["File"].apply(lambda x: Path(x).name)
    df_nav.set_index(["Name"], inplace=True)

    df_join = (
        df_nav.join(df_concat)
        .drop(
            columns=[
                "File",
                "Description",
                "Easting[m]",
                "Northing[m]",
                "Zone",
                "Date",
                "Time",
            ]
        )
        .reset_index()
    )

    # Create a GeoDataFrame
    df_join = gpd.GeoDataFrame(df_join, crs=CRS, geometry=df_join.geometry)

    # Output file paths
    outfile = dataset_path / f"{dataset_id}_footprints_full.gpkg"
    df_join.to_file(outfile, driver="GPKG")

    # Log successful output
    if verbosity >= 1:
        logging.info(f"Full footprints GeoPackage created at: {outfile}")

    # Dissolve geometries and create a new GeoDataFrame
    df_dissolved = gpd.GeoDataFrame(geometry=[df_join.unary_union], crs=CRS)
    df_dissolved["Dataset"] = dataset_id

    # Output for dissolved geometries
    outfile_dissolved = dataset_path / f"{dataset_id}_footprints_dissolved.gpkg"
    df_dissolved.to_file(outfile_dissolved, driver="GPKG")

    # Log successful output for dissolved geometries
    if verbosity >= 1:
        logging.info(f"Dissolved footprints GeoPackage created at: {outfile_dissolved}")

# Example usage:
# create_dataset_footprints(Path('/path/to/dataset'), verbosity=2)

