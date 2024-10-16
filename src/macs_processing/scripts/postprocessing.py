import argparse
import importlib
import logging
import sys

# ignore warnings
import warnings


from macs_processing.utils.processing import *
from macs_processing.utils.postprocessing import *
from macs_processing.utils.whiteboxtools import *

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--settings", required=True, type=Path, help="Path to Settings file"
)

parser.add_argument(
    "-dsm",
    "--dsm_mode",
    default="wbt",
    type=str,
    choices=["pix4d", "wbt"],
    help='Set dsm_processing_mode, "pix4d" for using pix4d provided dsm tiles; "wbt" for custom '
    "whiteboxtools created DSM",
)

parser.add_argument(
    "-pc",
    "--point_cloud",
    default="both",
    type=str,
    choices=["both", "nir", "rgb"],
    help='Set which point cloud to use. Options: "both", "nir", "rgb"',
)

parser.add_argument(
    "-m", "--mosaic", action="store_true", help="Set flag to calculate COG mosaic"
)

parser.add_argument(
    "-keep_dsm_if",
    "--keep_dsm_if",
    action="store_true",
    help="keep intermediate DSM files (for debugging)",
)

parser.add_argument(
    "--n_jobs", default=40, type=int, help="Set number of parallel processes"
)


args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###

# logger
logfile = settings.PROJECT_DIR / f"{settings.PROJECT_DIR.name}_postprocessing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
)

# ## Post Processing
# #### PostProcess Optical
# * create 4 band mosaic + calculate pyramids

logging.info("Start postprocessing Orthoimage tiles!")

PROCESS = True


def main():
    tiles_dir = (
        Path(settings.PROJECT_DIR)
        / "04_pix4d"
        / settings.PIX4d_PROJECT_NAME
        / "3_dsm_ortho"
        / "2_mosaic"
        / "tiles"
    )

    # Setup processing paths - make more consistent
    PRODUCT_DIR = Path(settings.PROJECT_DIR) / "06_DataProducts"
    settings.TARGET_DIR_ORTHO = PRODUCT_DIR / "Ortho"
    # temporary processing dirs
    TMP_DIR_ORTHO = PRODUCT_DIR / "tmp_mosaics"
    TMP_DIR_VRT = PRODUCT_DIR / "tmp_vrt"
    TMP_MASK_VECTORIZE_DIR = PRODUCT_DIR / "tmp_footprints"

    settings.TARGET_DIR_DSM = PRODUCT_DIR / "DSM"
    settings.TARGET_DIR_PC = PRODUCT_DIR / "PointClouds"

    tiles_dir = (
        Path(settings.PROJECT_DIR)
        / "04_pix4d"
        / settings.PIX4d_PROJECT_NAME
        / "3_dsm_ortho"
        / "2_mosaic"
        / "tiles"
    )

    # setup DSM file paths
    if args.dsm_mode == "pix4d":
        tiles_dir_dsm = (
            Path(settings.PROJECT_DIR)
            / "04_pix4d"
            / settings.PIX4d_PROJECT_NAME
            / "3_dsm_ortho"
            / "1_dsm"
            / "tiles"
        )
    else:
        tiles_dir_dsm = (
            Path(settings.PROJECT_DIR)
            / "04_pix4d"
            / settings.PIX4d_PROJECT_NAME
            / "3_dsm_ortho"
            / "1_dsm"
            / "tiles_wbt"
        )

    # get region properties, date and resolution
    region, site, site_number, date, resolution = parse_site_name_v2(settings.SITE_NAME)

    # create target/output dirs
    os.makedirs(settings.TARGET_DIR_ORTHO, exist_ok=True)
    os.makedirs(settings.TARGET_DIR_DSM, exist_ok=True)

    # get tile_ids
    flist = list(tiles_dir.glob("*.tif"))
    df = flist_to_df(flist)
    df["tile_id"] = df.apply(lambda x: x.row + "_" + x.col, axis=1)
    tiles = pd.unique(df["tile_id"])

    # Get sensor names
    nir_sensor = get_nir_sensor_name(df)
    rgb_sensor = get_rgb_sensor_name(df)

    # Validate input files - raise error if file sets are completely empty
    content_is_valid = check_ortho_validity(df)
    if not content_is_valid:
        raise Exception("Processed Orthotiles do not contain Data")

    #### Run
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(
            delayed(full_postprocessing_optical)(
                df,
                tile,
                nir_name=nir_sensor,
                rgb_name=rgb_sensor,
                vrt_dir=TMP_DIR_VRT,
                target_dir_mosaic=TMP_DIR_ORTHO,
            )
            for tile in tqdm.tqdm(tiles[:])
        )
        logging.info("Finished postprocessing Orthoimage tiles!")

    # #### Move and rename to output

    logging.info("Start moving and renaming Ortho tiles!")

    if True:
        flist = list(TMP_DIR_ORTHO.glob("mosaic*.tif"))
        df = flist_to_df(flist)
        df["tile_id"] = df.apply(lambda x: x.row + "_" + x.col, axis=1)

        # TODO: uncomment
        move_and_rename_processed_tiles(
            df, settings.SITE_NAME, settings.TARGET_DIR_ORTHO, "Ortho", move=True
        )
        logging.info("Finished moving and renaming Ortho tiles!")
    # """
    # #### DSM
    logging.info("Start moving and renaming DSM tiles!")
    # Check for processing mode
    if args.dsm_mode == "pix4d":
        logging.info("Creating DSMfile with original Pix4D output!")
        tiles_dir_dsm = (
            Path(settings.PROJECT_DIR)
            / "04_pix4d"
            / settings.PIX4d_PROJECT_NAME
            / "3_dsm_ortho"
            / "1_dsm"
            / "tiles"
        )

    else:
        logging.info("Creating DSMfile with WhiteBoxTools!")
        tiles_dir_dsm = (
            Path(settings.PROJECT_DIR)
            / "04_pix4d"
            / settings.PIX4d_PROJECT_NAME
            / "3_dsm_ortho"
            / "1_dsm"
            / "tiles_wbt"
        )
        # tiles_dir_dsm = PRODUCT_DIR / 'tmp_dsm'
        if tiles_dir_dsm.exists():
            # pass
            shutil.rmtree(tiles_dir_dsm)
        os.makedirs(tiles_dir_dsm, exist_ok=True)

        point_cloud_dir = (
            Path(settings.PROJECT_DIR)
            / "04_pix4d"
            / settings.PIX4d_PROJECT_NAME
            / "2_densification"
            / "point_cloud"
        )

        # temp_dir_dsm = Path(settings.PROJECT_DIR)
        wbt.set_working_dir(point_cloud_dir)

        # get region and file properties
        tile_index_list = list(tiles_dir.glob("*.tif"))
        crs = crs_from_file(tile_index_list[0])
        resolution = resolution_from_file(tile_index_list[0])

        # run processes
        if True:
            # Merge Point Clouds
            merged_pc = merge_point_clouds(which_point_cloud=args.point_cloud)
            # Interpolate Point Cloud to DSM
            merged_pc_IDW = pc_IDW_toDSM(infile=merged_pc, resolution=resolution)
            # Fill small holes
            merged_pc_IDW_filled = fill_holes(
                infile=merged_pc_IDW, filter=int(5 / resolution)
            )
            # Smooth DSM
            merged_pc_IDW_filled_smoothed = smooth_DSM(merged_pc_IDW_filled, filter=11)
            # Add Projection
            wbt_final_dsm_file = assign_crs_to_raster(
                merged_pc_IDW_filled_smoothed, crs=crs
            )

        if True:
            # tiling
            dsm_mosaic = point_cloud_dir / wbt_final_dsm_file
            _ = Parallel(n_jobs=args.n_jobs)(
                delayed(clip_to_tile)(dsm_mosaic, f, target_dir=tiles_dir_dsm)
                for f in tqdm.tqdm(tile_index_list[:])
            )

            # cleanup dsm mosaic and intermediate files
            if not args.keep_dsm_if:
                for file_delete in [
                    merged_pc,
                    merged_pc_IDW,
                    merged_pc_IDW_filled,
                    merged_pc_IDW_filled_smoothed,
                    wbt_final_dsm_file,
                ]:
                    try:
                        print("Delete temporary files!")
                        os.remove(point_cloud_dir / file_delete)
                    except:
                        continue
    # """
    flist_dsm = list(tiles_dir_dsm.glob("*.tif"))
    df_dsm = flist_to_df(flist_dsm)
    df_dsm["tile_id"] = df_dsm.apply(lambda x: x.row + "_" + x.col, axis=1)
    if True:
        move_and_rename_processed_tiles(
            df_dsm, settings.SITE_NAME, settings.TARGET_DIR_DSM, "DSM", move=False
        )
    logging.info("Finished moving and renaming DSM tiles!")

    # #### Create footprints file
    os.makedirs(TMP_MASK_VECTORIZE_DIR, exist_ok=True)
    # TODO: This is doing nothing
    Path(os.environ["CONDA_PREFIX"]) / "Scripts" / "gdal_polygonize.py"

    # make geopackage - perhaps in the end
    FOOTPRINTS_FILE = PRODUCT_DIR / f"{settings.SITE_NAME}_tile_footprints.geojson"

    logging.info(f"Start merging footprints to file {FOOTPRINTS_FILE}!")
    # create vector mask of Data (DN=0 for noData, DN=255 for valid Data)
    flist_out = list(settings.TARGET_DIR_ORTHO.glob("*.tif"))

    if True:
        vector_list = Parallel(n_jobs=args.n_jobs)(
            delayed(create_mask_vector)(infile, TMP_MASK_VECTORIZE_DIR)
            for infile in tqdm.tqdm(flist_out[:])
        )

    # Merge vectors and remove noData parts
    if True:
        gdf_list = Parallel(n_jobs=args.n_jobs)(
            delayed(load_and_prepare_footprints)(vector_file)
            for vector_file in tqdm.tqdm(vector_list[:])
        )

    merge_single_vector_files(gdf_list, FOOTPRINTS_FILE, settings.SITE_NAME, date)

    logging.info("Finished processing!")

    # remove NoData
    logging.info("Deleting empty tiles!")
    if True:
        delete_empty_product_tiles(
            FOOTPRINTS_FILE, settings.TARGET_DIR_ORTHO, settings.TARGET_DIR_DSM
        )

    # START DSM PROCESSING
    logging.info("Start postprocessing DSM tiles!")
    # Clip DSM to footprints
    df = gpd.read_file(FOOTPRINTS_FILE)
    fnames = df["DSM"]
    DSM_DIR_TMP = settings.TARGET_DIR_DSM.parent / "DSM_tmp"
    delete_input = True
    os.makedirs(DSM_DIR_TMP)

    if True:
        Parallel(n_jobs=1)(
            delayed(clip_dsm_to_bounds)(
                FOOTPRINTS_FILE, filename, settings.TARGET_DIR_DSM, DSM_DIR_TMP
            )
            for filename in tqdm.tqdm(fnames[:])
        )
        if delete_input:
            shutil.rmtree(settings.TARGET_DIR_DSM)
            os.rename(DSM_DIR_TMP, settings.TARGET_DIR_DSM)

    logging.info("Finished postprocessing DSM tiles!")

    # ############### CLIP Point Clouds to footprint ################### #
    # extract individual tiles from footprints file
    logging.info("Start tiling Point Clouds!")
    os.makedirs(settings.TARGET_DIR_PC, exist_ok=True)

    # point cloud
    point_clouds_dir = (
        settings.Path(settings.PROJECT_DIR)
        / "04_pix4d"
        / settings.PIX4d_PROJECT_NAME
        / "2_densification"
        / "point_cloud"
    )
    point_cloud_nir = list(
        point_clouds_dir.glob(f"*{nir_sensor}_densified_point_cloud.las")
    )[0]
    point_cloud_rgb = list(
        point_clouds_dir.glob(f"*{rgb_sensor}_densified_point_cloud.las")
    )[0]
    # RUN Point Cloud Clipping
    # NIR Point Cloud
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(
            delayed(create_point_cloud_tiles_las2las)(
                point_cloud_nir,
                tile,
                settings,
                target_dir=settings.TARGET_DIR_PC,
                product_name="PointCloudNIR",
            )
            for tile in tqdm.tqdm(vector_list[:])
        )
    # RGB Point Cloud
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(
            delayed(create_point_cloud_tiles_las2las)(
                point_cloud_rgb,
                tile,
                settings,
                target_dir=settings.TARGET_DIR_PC,
                product_name="PointCloudRGB",
            )
            for tile in tqdm.tqdm(vector_list[:])
        )

    logging.info("Finished tiling Point Clouds!")

    # Create VRT files
    working_dir = Path(os.getcwd())
    vrt_path = working_dir / "create_vrt.py"
    create_vrt(products_dir=PRODUCT_DIR, vrt_script_location=vrt_path)
    os.chdir(working_dir)

    # create previews
    logging.info("Creating previews!")
    create_previews(products_dir=PRODUCT_DIR, pyramid_level=1, overwrite=True)

    # """
    # create COG mosaics
    if args.mosaic:
        logging.info("Create COG mosaics!")
        ortho_vrt = PRODUCT_DIR / "Ortho.vrt"
        ortho_cog = PRODUCT_DIR / f"{settings.SITE_NAME}_Ortho.tif"
        dsm_vrt = PRODUCT_DIR / "DSM.vrt"
        dsm_cog = PRODUCT_DIR / f"{settings.SITE_NAME}_DSM.tif"
        hillshade_cog = PRODUCT_DIR / f"{settings.SITE_NAME}_Hillshade.tif"
        s_cog_ortho = f"gdal_translate -stats -of COG -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE {ortho_vrt} {ortho_cog}"
        s_cog_dsm = f"gdal_translate -stats -of COG -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE {dsm_vrt} {dsm_cog}"
        s_hillshade = f"gdaldem hillshade -multidirectional -of COG -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=DEFLATE {dsm_cog} {hillshade_cog}"
        for run in [s_cog_ortho, s_cog_dsm, s_hillshade]:
            os.system(run)

    # Copy processing report, nav file log file
    logging.info("Copying reports!")
    processing_info_dir = PRODUCT_DIR / "processing_info"
    os.makedirs(processing_info_dir, exist_ok=True)

    report_file = (
        Path(settings.PROJECT_DIR)
        / "04_pix4d"
        / settings.SITE_NAME
        / "1_initial"
        / "report"
        / f"{settings.PIX4d_PROJECT_NAME}_report.pdf "
    )
    if report_file.exists():
        shutil.copy(report_file, processing_info_dir)

    logging.info("Processing navfile!")
    nav_file_in = (
        Path(settings.PROJECT_DIR) / "01_rawdata" / "tif" / "geo_pix4d_new.txt"
    )
    nav_file_out = processing_info_dir / f"{settings.SITE_NAME}_nav.txt"

    try:
        shutil.copy(nav_file_in, nav_file_out)
    except:
        print("nav_file_in could not be found! Copy skipped")
    shutil.copy(logfile, processing_info_dir)

    # Cleanup temporary dirs
    logging.info("Deleting temporary files and directories!")
    # delete pyramids
    # DSM
    overview_dsm = PRODUCT_DIR / "DSM.vrt.ovr"
    if overview_dsm.exists():
        os.remove(overview_dsm)
    # Ortho
    overview_ortho = PRODUCT_DIR / "Ortho.vrt.ovr"
    if overview_ortho.exists():
        os.remove(overview_ortho)

    shutil.rmtree(TMP_MASK_VECTORIZE_DIR)
    shutil.rmtree(TMP_DIR_ORTHO)
    shutil.rmtree(TMP_DIR_VRT)

    logging.info("Finished Postprocessing!")


if __name__ == "__main__":
    main()
