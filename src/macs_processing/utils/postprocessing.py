import os
import shutil
import tarfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import tqdm
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal, ogr
from sklearn import preprocessing


def flist_to_df(filelist):
    """
    Create pandas DataFrame with information parsed from filelist

    Parameters
    ----------
    filelist : list
        file list of MACS Orthomosaic Output files

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    sensors = []
    rows = []
    cols = []
    for f in filelist:
        sensor, row, col = f.name[:-4].split("_")[-3:]
        sensors.append(sensor)
        rows.append(row)
        cols.append(col)

    df = pd.DataFrame(columns=["filename", "sensor", "row", "col"])
    df["filename"] = filelist
    df["sensor"] = sensors
    df["row"] = rows
    df["col"] = cols
    return df


# create more specific vrt files for each run to avoid duplicates on parallel run
def stack_output(
    outmosaic: str | Path,
    rgbfile: str | Path,
    nirfile: str | Path,
    remove_temporary_files: bool = True,
    vrt_dir: Path = Path("."),
):
    """
    Stack RGB and NIR images into a single Cloud Optimized GeoTIFF (COG).

    This function reads a 4-band RGB image (RGB-A) and a 2-band NIR image (NIR-A),
    creates virtual raster datasets (VRTs) for each band, and then combines them
    into a single COG output file. The function can also remove temporary files
    created during the process.

    Parameters:
        outmosaic (str or Path): The path where the output COG file will be saved.
        rgbfile (str or Path): The path to the input RGB image file.
        nirfile (str or Path): The path to the input NIR image file.
        remove_temporary_files (bool): If True, temporary VRT files will be deleted
                                        after the output is created. Default is True.
        vrt_dir (Path): The directory where temporary VRT files will be created.
                        Default is the current directory.

    Returns:
        None: This function does not return any value. It writes the output file
              directly to the specified location.
    """
    vrt_dir.mkdir(exist_ok=True)
    basename_rgb = rgbfile.name[:-4]
    basename_nir = nirfile.name[:-4]

    mos = vrt_dir / f"{basename_rgb}.vrt"
    rgb_vrt = []

    # Open RGB bands and create VRTs
    rgb_dataset = gdal.Open(rgbfile, gdal.GA_ReadOnly)
    for band in [1, 2, 3]:
        infile = vrt_dir / f"{basename_rgb}_{band}.vrt"
        gdal.Translate(str(infile), rgb_dataset, bandList=[band])
        rgb_vrt.append(str(infile))

    # Open NIR band and create a VRT
    infile_nir = vrt_dir / f"{basename_nir}_1.vrt"
    nir_dataset = gdal.Open(nirfile, gdal.GA_ReadOnly)
    gdal.Translate(str(infile_nir), nir_dataset, bandList=[1])

    # Build the final VRT for RGB and NIR bands
    gdal.BuildVRT(str(mos), rgb_vrt + [str(infile_nir)], separate=True)

    # Translate the final VRT to Cloud Optimized GeoTIFF format
    gdal.Translate(
        str(outmosaic),
        str(mos),
        format="GTiff",
        noData=0,
        creationOptions=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "BIGTIFF=YES",
            "COG=YES",
            "IGNORE_COG_LAYOUT_BREAK",
        ],
    )

    # Clean up temporary files if needed
    if remove_temporary_files:
        for file in rgb_vrt + [str(infile_nir), str(mos)]:
            os.remove(file)

    # Clean up datasets
    rgb_dataset = None
    nir_dataset = None


def mask_and_name_bands(mosaic_file):
    """
    Function to mask incomplete spectral data (e.g. with only NIR data and no RGB and vice versa)
    Add names to Bands
    """
    with rasterio.open(
        mosaic_file, "r+", options={"IGNORE_COG_LAYOUT_BREAK": "YES"}
    ) as src:
        src.profile["nodata"] = 0
        data = src.read()
        newmask = ~(data == 0).any(axis=0)
        newmask_write = np.r_[src.count * [newmask]]
        data_masked = data * newmask_write
        src.set_band_description(1, "MACS Blue Band")
        src.set_band_description(2, "MACS Green Band")
        src.set_band_description(3, "MACS Red Band")
        src.set_band_description(4, "MACS NIR Band")
        src.write(data_masked)


def get_nir_sensor_name(df):
    for sensor_name in ["nir", "grayscale"]:
        if len(df.query(f'sensor=="{sensor_name}"')) > 0:
            return sensor_name
    else:
        raise NameError("sensor name for NIR band does not match")


def get_rgb_sensor_name(df):
    for sensor_name in ["rgb", "group1", "RGB"]:
        if len(df.query(f'sensor=="{sensor_name}"')) > 0:
            return sensor_name
    else:
        raise NameError("sensor name for NIR band does not match")


def check_tile_validity(image_path):
    with rasterio.open(image_path, options={"IGNORE_COG_LAYOUT_BREAK": "YES"}) as src:
        return src.read(1).mean() != 0


def check_ortho_validity(df, n_jobs=40):
    """
    This function checks the validity of both NIR (Near-Infrared) and RGB (Red-Green-Blue) images in a given dataframe.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the filenames and sensor types of the images.
    n_jobs (int, optional): The number of jobs to run in parallel. Default is 40.

    The DataFrame 'df' is expected to have at least two columns: 'sensor' and 'filename'.
    The 'sensor' column should contain the sensor type of the image ('nir' or 'rgb'),
    and the 'filename' column should contain the filename of the image.

    The function uses the 'check_tile_validity' function (not defined in this scope) to check the validity of each image.
    This is done in parallel using the joblib library's 'Parallel' and 'delayed' functions.

    After checking the validity of the images, the function prints the number of valid NIR and RGB images,
    and whether there is any content in them. It also prints whether postprocessing should continue based on the validity of the images.

    Returns:
    bool: True if all images are valid, False otherwise.
    """

    # test for nir validity
    flist_nir = df.query('sensor == "nir"')["filename"].values
    nir_validity = Parallel(n_jobs=n_jobs)(
        delayed(check_tile_validity)(im) for im in flist_nir[:]
    )
    # test for rgb validity
    flist_rgb = df.query('sensor == "rgb"')["filename"].values
    rgb_validity = Parallel(n_jobs=n_jobs)(
        delayed(check_tile_validity)(im) for im in flist_rgb[:]
    )

    # Validity for entire subset, needs to have at least one tile with data
    nir_valid = any(nir_validity)
    rgb_valid = any(rgb_validity)

    # documentation
    print(
        f"NIR images have content:{nir_valid}, {sum(nir_validity)}/{len(flist_nir)} images"
    )
    print(
        f"RGB images have content:{rgb_valid}, {sum(rgb_validity)}/{len(flist_rgb)} images"
    )
    is_valid = nir_valid & rgb_valid
    print(f"Continue Postprocessing: {is_valid}")

    return is_valid


def full_postprocessing_optical(
    df,
    tile_id,
    rgb_name="group1",
    nir_name="nir",
    target_dir_mosaic=None,
    vrt_dir=Path("."),
):
    """
    Wrapper function to sequentially run

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    tile_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    subset = df[df["tile_id"] == tile_id]
    rgbfile = subset.query(f'sensor=="{rgb_name}"').filename.values[0]
    nirfile = subset.query(f'sensor=="{nir_name}"').filename.values[0]
    if target_dir_mosaic is not None:
        target_dir_mosaic.mkdir(exist_ok=True)
        outmosaic = target_dir_mosaic / f"mosaic_{tile_id}.tif"
    else:
        outmosaic = rgbfile.parent / f"mosaic_{tile_id}.tif"
    stack_output(
        outmosaic, rgbfile, nirfile, remove_temporary_files=True, vrt_dir=vrt_dir
    )
    mask_and_name_bands(outmosaic)


def move_and_rename_processed_tiles(
    df, out_basename, target_dir, product_type, move=True
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe created by flist_to_df().
    out_basename : str
        basename of outputfile.
    target_dir : str/Path
        output directory.
    product_type : str
        simple name identifier e.g. 'Ortho' or 'DSM'.
    move : bool, optional
        set True if files should be moved instead of copied. The default is False.

    Returns
    -------
    None.

    """
    for index, row in df.iloc[:].iterrows():
        tile_id = row.tile_id
        infile = row.filename
        outfile = target_dir / f"{out_basename}_{product_type}_{tile_id}.tif"

        if move:
            shutil.move(infile, outfile)
        else:
            shutil.copy(infile, outfile)

        """
        try:

            if move:
                shutil.move(infile_ovr, outfile_ovr)
            else:
                shutil.copy(infile_ovr, outfile_ovr)
        
        except Exception as e:
            print(e)
        """


def create_mask_vector(
    raster_file: Path, temporary_target_dir: Path, remove_raster_mask: bool = False
) -> Path:
    """
    Create a vector mask from a raster file.

    Parameters
    ----------
    raster_file : Path
        Input raster file from which to create the vector mask.
    temporary_target_dir : Path
        Directory where temporary files will be stored.
    remove_raster_mask : bool, optional
        If True, the temporary raster mask will be deleted. Default is False.

    Returns
    -------
    Path
        Path of the output vector mask file in GeoJSON format.
    """

    # Ensure the temporary directory exists
    temporary_target_dir.mkdir(parents=True, exist_ok=True)

    # Define paths for the mask raster and output vector
    maskfile = temporary_target_dir / (raster_file.stem + "_mask.tif")
    mask_vector = temporary_target_dir / (raster_file.stem + "_mask.geojson")

    # Open the raster file and create a mask
    raster_dataset = gdal.Open(str(raster_file), gdal.GA_ReadOnly)
    
    if raster_dataset is None:
        raise FileNotFoundError(f"Unable to open raster file: {raster_file}")

    # Create a mask from the raster dataset
    gdal.Translate(
        str(maskfile),
        raster_dataset,
        format="GTiff",
        outputType=gdal.GDT_Byte,
        bandList=[1],
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )

    # Open the created mask file to ensure it's valid
    mask_dataset = gdal.Open(str(maskfile), gdal.GA_ReadOnly)
    
    if mask_dataset is None:
        raise FileNotFoundError(f"Unable to open mask file: {maskfile}")

    # Polygonize the mask raster into a vector format
    driver = ogr.GetDriverByName("GeoJSON")
    
    if driver is None:
        raise RuntimeError("GeoJSON driver not available.")

    # Create an empty vector data source for the output GeoJSON
    vector_data_source = driver.CreateDataSource(str(mask_vector))
    
    if vector_data_source is None:
        raise RuntimeError(f"Failed to create vector data source: {mask_vector}")

    # Create a layer for the vector data source
    layer = vector_data_source.CreateLayer("mask", geom_type=ogr.wkbPolygon)
    
    if layer is None:
        raise RuntimeError("Failed to create layer in vector data source.")

    # Use gdal.Polygonize to convert the mask raster to polygons
    band = mask_dataset.GetRasterBand(1)
    
    if band is None:
        raise RuntimeError("Failed to get band from mask dataset.")

    gdal.Polygonize(band, None, layer, -1, [], callback=None)

    # Clean up datasets
    raster_dataset = None
    mask_dataset = None  # Close the mask dataset properly
    vector_data_source = None  # This will automatically close and save changes

    # Optionally remove the temporary raster mask
    if remove_raster_mask:
        maskfile.unlink()  # Use unlink() to remove the file

    return mask_vector


def load_and_prepare_footprints(vector_file):
    """
    Parameters
    ----------
    vector_file : Path
        OGR compatible vector file (basic footprint).

    Returns
    -------
    gdf : TYPE
        filled GeoDataframe.

    """
    gdf = gpd.read_file(vector_file)
    file_name_raster = Path(str(vector_file).replace("_mask.geojson", ".tif")).name
    gdf["Orthomosaic"] = file_name_raster
    gdf["DSM"] = file_name_raster.replace("_Ortho_", "_DSM_")
    return gdf


def merge_single_vector_files(gdf_list, outfile, site_name, date_local):
    """
    Function to merge a list of Geodataframes (individual image footprints) \
    to a larger one.
    
    Parameters
    ----------
    gdf_list : list
        List of GeoDataframes of identical type.
    outfile : Path
        path of merged GeoDataframe.
    site_name : str
        basename of site identifier.
    date_local : str
        date in 'YYYY-MM-DD' format.

    Returns
    -------
    None.

    """
    # Merge and save final footprints file
    gdf_merged = gpd.GeoDataFrame(pd.concat(gdf_list))
    gdf_merged = gdf_merged.set_crs(crs=gdf_list[0].crs).to_crs(epsg=4326)

    gdf_merged = gdf_merged[gdf_merged["DN"] > 0]
    gdf_merged = gdf_merged.drop(columns=["DN"])

    gdf_merged["Site_name"] = site_name
    gdf_merged["Date"] = date_local

    # dissolve single part to multipart
    cols = gdf_merged.columns
    gdf_merged = gdf_merged.dissolve(by="Orthomosaic").reset_index(drop=False)[cols]

    gdf_merged.to_file(outfile)


def delete_empty_product_tiles(footprints_file, Orthodir, DSMdir):
    """

    Parameters
    ----------
    footprints_file : Path, str
        Path to product footprints file.
    Orthodir : Path
        Path to Ortho products dir.
    DSMdir : Path
        Path to DSM products dir.

    Returns
    -------
    None.

    """

    df = gpd.read_file(footprints_file)

    flist_ortho = list(Orthodir.glob("*.tif"))
    delete_ortho = [f for f in flist_ortho if f.name not in df["Orthomosaic"].values]
    for f in delete_ortho[:]:
        try:
            os.remove(f)
        except:
            print(f"Skipped deleting {f.name}")

    flist_dsm = list(DSMdir.glob("*.tif"))
    delete_dsm = [f for f in flist_dsm if f.name not in df["DSM"].values]
    for f in delete_dsm[:]:
        try:
            os.remove(f)
        except:
            print(f"Skipped deleting {f.name}")


def parse_site_name(site_name):
    region, site, site_number, date_tmp, resolution = site_name.split("_")
    date = f"{date_tmp[:4]}-{date_tmp[4:6]}-{date_tmp[6:]}"
    return region, site, site_number, date, resolution


def parse_site_name_v2(site_name):
    region, site, date_tmp, resolution, site_number = site_name.split("_")
    date = f"{date_tmp[:4]}-{date_tmp[4:6]}-{date_tmp[6:]}"
    return region, site, site_number, date, resolution


def clip_dsm_to_bounds(footprints_file: Path, filename: str, dsmdir: Path, outdir: Path) -> None:
    """
    Clip a DSM (Digital Surface Model) raster to the bounds defined by footprints.

    Parameters
    ----------
    footprints_file : Path
        Path to the vector file containing footprint geometries.
    filename : str
        Name of the DSM raster file to be clipped.
    dsmdir : Path
        Directory containing the input DSM files.
    outdir : Path
        Directory where the clipped output file will be saved.

    Returns
    -------
    None
    """
    
    # Define input and output file paths
    infile = dsmdir / filename
    outfile = outdir / filename

    # Open the raster dataset
    raster_dataset = gdal.Open(str(infile), gdal.GA_ReadOnly)
    
    # Open the vector dataset (footprints)
    vector_dataset = ogr.Open(str(footprints_file))
    
    # Create a mask from the footprints based on the specified condition
    layer = vector_dataset.GetLayer()
    
    # Create options for gdal.Warp to clip the raster using the vector layer
    options = gdal.WarpOptions(cutlineLayer=layer, 
                                cropToCutline=True, 
                                format='GTiff', 
                                creationOptions=['COMPRESS=DEFLATE'])

    # Perform the warping operation (clipping)
    gdal.Warp(str(outfile), raster_dataset, options=options)

    # Clean up datasets
    raster_dataset = None
    vector_dataset = None


def prepare_band(inband, noData=0, p_low=0, p_high=98):
    mask = inband == noData
    p2 = np.percentile(inband[~mask], p_low)
    p98 = np.percentile(inband[~mask], p_high)
    normed = np.clip(
        preprocessing.MinMaxScaler().fit_transform(np.clip(inband, p2, p98)), 0, 1
    )
    normed[mask] = 0
    return np.ma.masked_where(mask, normed)


def show_3band_image(image, savepath=None, **kwargs):
    fig, ax = plt.subplots(dpi=300, **kwargs)
    ax.imshow(image)
    ax.set_xticklabels("")
    ax.set_xticks([])
    ax.set_yticklabels("")
    ax.set_yticks([])
    if savepath:
        fig.savefig(savepath)


def show_dsm_image(
    dsm, savepath=None, noData=[-10000, -32768], show_colorbar=False, **kwargs
):
    # mask no data
    if not isinstance(noData, list):
        noData = [noData]
    mask = np.isin(dsm, [-10000, -32768])

    # calculate image stats for visualization
    p_low = np.percentile(dsm[~mask], 1)
    if p_low < -5:
        p_low = -5
    p_high = np.percentile(dsm[~mask], 99)

    # create figure
    fig, ax = plt.subplots(dpi=300, **kwargs)
    im = ax.imshow(
        np.ma.masked_where(mask, dsm), vmin=p_low, vmax=p_high, cmap=plt.cm.terrain
    )
    ax.set_xticklabels("")
    ax.set_xticks([])
    ax.set_yticklabels("")
    ax.set_yticks([])

    # add colorbar
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    if savepath:
        fig.savefig(savepath)


def load_ortho(image_path, pyramid_level=-2, overviews=[2, 4, 8]):
    with rasterio.open(
        image_path, "r+", options={"IGNORE_COG_LAYOUT_BREAK": "YES"}
    ) as src:
        src.build_overviews(overviews)
        oviews = src.overviews(1)  # list of overviews from biggest to smallest
        oview = oviews[pyramid_level]  # Use second-highest lowest overview
        print("Decimation factor= {}".format(oview))
        red = src.read(
            3, out_shape=(1, int(src.height // oview), int(src.width // oview))
        )
        green = src.read(
            2, out_shape=(1, int(src.height // oview), int(src.width // oview))
        )
        blue = src.read(
            1, out_shape=(1, int(src.height // oview), int(src.width // oview))
        )
        nir = src.read(
            4, out_shape=(1, int(src.height // oview), int(src.width // oview))
        )
    return blue, green, red, nir


def load_dsm(image_path, pyramid_level=-2, overviews=[2, 4, 8]):
    with rasterio.open(
        image_path, "r+", options={"IGNORE_COG_LAYOUT_BREAK": "YES"}
    ) as src:
        src.build_overviews(overviews)
        oviews = src.overviews(1)  # list of overviews from biggest to smallest
        oview = oviews[pyramid_level]  # Use second-highest lowest overview
        print("Decimation factor= {}".format(oview))
        dsm = src.read(
            1, out_shape=(1, int(src.height // oview), int(src.width // oview))
        )
    return dsm


def prepare_image_3band(band_list):
    rgb_list = [prepare_band(band) for band in band_list]
    rgb_list.append(~rgb_list[0].mask)
    rgb_image = np.dstack(rgb_list)
    return rgb_image


def create_vrt(products_dir, vrt_script_location):
    """ """
    # create vrt
    vrt_script = Path(vrt_script_location).name
    shutil.copy(vrt_script_location, products_dir)
    os.chdir(products_dir)
    os.system(f"python {vrt_script}")
    os.remove(vrt_script)


def create_previews(products_dir, overwrite=False, pyramid_level=-2):
    """
    full process wrapper
    """
    # check if output files already exist
    basename = products_dir.parent.name
    filename_rgb = products_dir / f"{basename}_preview_RGB.png"
    filename_cir = products_dir / f"{basename}_preview_CIR.png"
    filename_dsm = products_dir / f"{basename}_preview_DSM.png"

    fname_exist = [
        (products_dir / fname).exists()
        for fname in [filename_rgb, filename_cir, filename_dsm]
    ]
    if np.all(fname_exist):
        print("files already exist!")
        if not overwrite:
            print("Skipped processing!")
            return 0
        else:
            print("Overwriting output files!")

    print("Start processing previews")

    # Load Ortho + dsm
    blue, green, red, nir = load_ortho(
        products_dir / "Ortho.vrt", pyramid_level=pyramid_level
    )
    dsm = load_dsm(products_dir / "DSM.vrt", pyramid_level=pyramid_level)

    # prepare data
    # RGB
    rgb_image = prepare_image_3band([red, green, blue])
    cir_image = prepare_image_3band([nir, red, green])

    # show and save images
    show_3band_image(rgb_image, savepath=filename_rgb)
    show_3band_image(cir_image, savepath=filename_cir)
    show_dsm_image(dsm, savepath=filename_dsm, show_colorbar=True)


def create_unziplist2(
    flist,
    subset=["01_rawdata", "02_studysites", "04_pix4d"],
    exclude=["2_densification/", "3_dsm_ortho/"],
):
    outlist = []
    for sub in subset:
        # outlist.extend([f for f in flist if sub in f])
        outlist.extend(list(pd.Series(flist)[pd.Series(flist).str.contains(sub)]))
        # exclusion - not working yet
    if len(exclude) > 0:
        pattern = "|".join(exclude)
        r = pd.Series(outlist)
        contains = r.str.contains(pattern)
        outlist = r[~contains].values

    return outlist


def unzip_tarfile(p_file, site_name, target_dir):
    with tarfile.open(p_file) as f:
        flist = f.getnames()
        print(f"Number of files in archive: {len(flist)}")
        # check if rawdata should be processed
        subset = [
            f"04_pix4d/{site_name}/2_densification/point_cloud",
            "tile_footprints",
        ]
        unzip_subset = create_unziplist2(flist, subset=subset, exclude=[])
        assert len(unzip_subset) > 0
        unzip_subset = [f for f in unzip_subset if not (target_dir / f).exists()]
        print(f"Number of files to extract: {len(unzip_subset)}")
        # extract
        print("Start extraction to:", target_dir)
        for ff in tqdm.tqdm(unzip_subset[:]):
            f.extract(ff, path=target_dir)


def check_las2las_exists():
    """
    Check if the 'las2las' command exists in the system PATH.

    This function uses shutil.which() to search for the 'las2las' executable
    in the directories listed in the system's PATH environment variable.

    Raises:
        FileNotFoundError: If the 'las2las' command is not found in the system PATH.

    Returns:
        None

    Example:
        >>> check_las2las_exists()
        # If 'las2las' exists, function completes silently.
        # If 'las2las' doesn't exist, it raises a FileNotFoundError.

    Note:
        This function is typically used before attempting to execute the 'las2las'
        command to ensure its availability and provide a clear error message if
        the command is not installed or not in the PATH.
    """
    if shutil.which("las2las") is None:
        raise FileNotFoundError(
            "The 'las2las' command is not found in the system PATH."
        )
