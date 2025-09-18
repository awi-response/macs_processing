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


def load_kml_v2324(file, sensor="RGB", layer=0, sensor_name="111498_RGB"):
    """
    sensor_names: 99683_NIR 111498_RGB 1284123_TIR
    """
    df = gpd.read_file(file, driver="KML", layer=layer)
    df["Sensor"] = sensor
    df["Sensor Name"] = sensor_name
    df = df.set_index("Name", drop=True)
    return df


def validate_content(input_dir: Path) -> pd.Series:
    """
    Checks for the existence of expected KML and navigation files in a project directory.

    Parameters:
    - input_dir (Path): Path to the project directory to validate.

    Returns:
    - pd.Series: A pandas Series with:
        - 'project': The project directory name.
        - 'navfile': Boolean, True if the navigation file exists.
        - 'kml': Boolean, True if the KML file exists.
    """
    name = input_dir.name
    kml = (input_dir / f"{name}.kml").exists()
    nav = (input_dir / f"{name}_nav.txt").exists()
    return pd.Series(data=[name, nav, kml], index=["project", "navfile", "kml"])


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_dataset_footprints(
    dataset_path: Path,
    CRS: str = "EPSG:4326",
    regex_nav: str = "*nav.txt",
    verbosity: int = 0,
    force_overwrite: bool = False,
    year: int = 2024,
):
    """
    Processes KML files and generates GeoPackage outputs.

    This function reads KML files and associated navigation data, processes them into
    a GeoDataFrame, and outputs two GeoPackage files: one for full footprints and
    another for dissolved footprints. The function can handle different KML loading
    functions based on the specified year.

    Parameters:
    - dataset_path (Path): The path to the dataset folder containing KML and navigation files.
    - CRS (str): The Coordinate Reference System to be used for GeoDataFrames. Default is 'EPSG:4326'.
    - regex_nav (str): The regex pattern for navigation files. Default is '*nav.txt'.
    - verbosity (int): Level of verbosity for logging.
        0 for quiet,
        1 for project name only,
        2 for full details.
    - force_overwrite (bool): If True, overwrite existing output files. Default is False.
    - year (int): The year to determine which KML loading function to use.
        Supports special handling for 2023 and 2024 with specific loading functions.

    Returns:
    - None

    Notes:
    - If the output files already exist and `force_overwrite` is False, the function will exit without processing.
    - If `force_overwrite` is True, existing output files will be deleted before creating new ones.
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

    # Define output file paths
    outfile = dataset_path / f"{dataset_id}_footprints_full.gpkg"
    outfile_dissolved = dataset_path / f"{dataset_id}_footprints_dissolved.gpkg"

    # Check if output files already exist
    if outfile.exists() and outfile_dissolved.exists():
        if not force_overwrite:
            logging.info(
                f"Output files already exist: {outfile} and {outfile_dissolved}."
            )
            return  # Exit the function if both output files exist
        else:
            logging.info(
                f"Deleting existing output files: {outfile} and {outfile_dissolved}."
            )
            outfile.unlink()  # Delete the full footprints file
            outfile_dissolved.unlink()  # Delete the dissolved footprints file

    # Check individual output file existence based on force_overwrite
    if not force_overwrite:
        if outfile.exists():
            logging.info(f"Output file already exists: {outfile}.")
            return  # Exit the function if the full footprints file exists

        if outfile_dissolved.exists():
            logging.info(f"Dissolved output file already exists: {outfile_dissolved}.")
            return  # Exit the function if the dissolved footprints file exists

    # Load and concatenate data from KML files

    if year in [2023, 2024]:
        df_concat = gpd.GeoDataFrame(
            pd.concat(
                [
                    load_kml_v2324(
                        file_kml[0], sensor="RGB", layer=1, sensor_name="RGB 111498"
                    ),
                    load_kml_v2324(
                        file_kml[0], sensor="NIR", layer=0, sensor_name="NIR 99683"
                    ),
                    load_kml_v2324(
                        file_kml[0], sensor="TIR", layer=2, sensor_name="TIR 1284124"
                    ),
                ]
            ),
            crs=CRS,
        )
    elif year in [2025]:
        df_concat = gpd.GeoDataFrame(
            pd.concat(
                [
                    load_kml_v2324(
                        file_kml[0], sensor="RGB", layer=1, sensor_name="RGB 111498"
                    ),
                    load_kml_v2324(
                        file_kml[0], sensor="NIR", layer=0, sensor_name="NIR 99683"
                    ),
                    load_kml_v2324(
                        file_kml[0], sensor="TIR", layer=2, sensor_name="TIR 1284123"
                    ),
                ]
            ),
            crs=CRS,
        )
    else:
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

    # df_join = (
    #     df_nav.join(df_concat)
    #     .drop(
    #         columns=[
    #             "File",
    #             "Description",
    #             "Easting[m]",
    #             "Northing[m]",
    #             "Zone",
    #             "Date",
    #             "Time",
    #         ]
    #     )
    #     .reset_index()
    # )

    df_join = df_nav.join(df_concat).reset_index()
    # Create a GeoDataFrame
    df_join = gpd.GeoDataFrame(df_join, crs=CRS, geometry=df_join.geometry)

    # Output full footprints GeoPackage
    df_join.to_file(outfile, driver="GPKG")

    # Log successful output
    if verbosity >= 1:
        logging.info(f"Full footprints GeoPackage created at: {outfile}")

    # Dissolve geometries and create a new GeoDataFrame
    df_dissolved = gpd.GeoDataFrame(geometry=[df_join.union_all()], crs=CRS)
    df_dissolved["Dataset"] = dataset_id

    # Output for dissolved geometries
    df_dissolved.to_file(outfile_dissolved, driver="GPKG")

    # Log successful output for dissolved geometries
    if verbosity >= 1:
        logging.info(f"Dissolved footprints GeoPackage created at: {outfile_dissolved}")


# Example usage:
# create_dataset_footprints(Path('/path/to/dataset'), verbosity=2)
