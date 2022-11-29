import argparse
import importlib
import logging
import sys
# ignore warnings
import warnings

from processing_utils import *
from utils_postprocessing import *
from utils_wbt import *

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", default=Path('MACS_00_Settings.py'),
                    type=Path,
                    help="Path to Settings file")

parser.add_argument("-dsm", "--dsm_mode", default='pix4d', type=str, help='Set dsm_processing_mode, "pix4d" for using '
                                                                          'pix4d provided dsm tiles; "wbt" for custom '
                                                                          'whiteboxtools created DSM')
parser.add_argument("-pc", "--point_cloud", default='both', type=str,
                    help='Set which point cloud to use. Options: "both", "nir", "rgb"')
args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###

# logger
logfile = settings.PROJECT_DIR / f'{settings.PROJECT_DIR.name}.log'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(logfile),
                              logging.StreamHandler(sys.stdout)
                              ])

# ## Post Processing 
# #### PostProcess Optical
# * create 4 band mosaic + calculate pyramids

logging.info('Start postprocessing Orthoimage tiles!')


def main():
    tiles_dir = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
    flist = list(tiles_dir.glob('*.tif'))
    df = flist_to_df(flist)
    df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)
    tiles = pd.unique(df['tile_id'])

    # Get sensor names
    nir_sensor = get_nir_sensor_name(df)

    #### Run 
    # """
    _ = Parallel(n_jobs=40)(
        delayed(full_postprocessing_optical)(df, tile, nir_name=nir_sensor) for tile in tqdm.tqdm_notebook(tiles[:]))
    # """
    logging.info('Finished postprocessing Orthoimage tiles!')

    # #### Rename
    PRODUCT_DIR = Path(settings.PROJECT_DIR) / '06_DataProducts'
    settings.TARGET_DIR_ORTHO = PRODUCT_DIR / 'Ortho'
    settings.TARGET_DIR_DSM = PRODUCT_DIR / 'DSM'

    region, site, site_number, date, resolution = parse_site_name(settings.SITE_NAME)
    os.makedirs(settings.TARGET_DIR_ORTHO, exist_ok=True)
    os.makedirs(settings.TARGET_DIR_DSM, exist_ok=True)

    # #### Move and rename to output 

    logging.info('Start moving and renaming Ortho tiles!')

    tiles_dir = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
    flist = list(tiles_dir.glob('mosaic*.tif'))
    df = flist_to_df(flist)
    df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)

    # TODO: uncomment
    move_and_rename_processed_tiles(df, settings.SITE_NAME, settings.TARGET_DIR_ORTHO, 'Ortho', move=True)
    logging.info('Finished moving and renaming Ortho tiles!')

    # #### DSM 

    logging.info('Start moving and renaming DSM tiles!')
    # Change here to WBT based processing
    if args.dsm_mode == 'pix4d':
        tiles_dir_dsm = Path(
            settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles'

    else:
        logging.info('Creating DSMfile with WhiteBoxTools!')
        tiles_dir_dsm = Path(
            settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles_wbt'
        os.makedirs(tiles_dir_dsm, exist_ok=True)

        point_cloud_dir = Path(
            settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '2_densification' / 'point_cloud'

        # temp_dir_dsm = Path(settings.PROJECT_DIR)
        wbt.set_working_dir(point_cloud_dir)

        # TODO: check if it can be removed
        # ---------------------------
        wbt.set_compress_rasters(True)

        def my_callback(value):
            if not "%" in value:
                print(value)

        wbt.set_default_callback(my_callback)
        # ---------------------------
        # get region and file properties
        tile_index_list = list(tiles_dir.glob('*transparent_mosaic_group1*.tif'))
        crs = crs_from_file(tile_index_list[0])
        resolution = resolution_from_file(tile_index_list[0])

        # run processes
        # Merge Point Clouds
        merged_pc = merge_point_clouds(which_point_cloud=args.point_cloud)
        # Interpolate Point Cloud to DSM
        merged_pc_IDW = pc_IDW_toDSM(infile=merged_pc, resolution=resolution)
        # Fill small holes
        merged_pc_IDW_filled = fill_holes(infile=merged_pc_IDW, filter=int(5 / resolution))
        # Smooth DSM
        merged_pc_IDW_filled_smoothed = smooth_DSM(merged_pc_IDW_filled, filter=11)
        # Add Projection
        wbt_final_dsm_file = assign_crs_to_raster(merged_pc_IDW_filled_smoothed, crs=crs)

        # Cleanup
        for file_delete in [merged_pc, merged_pc_IDW, merged_pc_IDW_filled]:
            print('Delete temporary files!')
            os.remove(point_cloud_dir / file_delete)

        # wbt_final_dsm_file = 'merged_nir_IDW_filled_smoothed_projected.tif'
        # tiling
        dsm_mosaic = point_cloud_dir / wbt_final_dsm_file
        _ = Parallel(n_jobs=40)(
            delayed(clip_to_tile)(dsm_mosaic, f, target_dir=tiles_dir_dsm) for f in tqdm.tqdm(tile_index_list[:]))
        # cleanup dsm mosaic
        os.remove(dsm_mosaic)

    flist_dsm = list(tiles_dir_dsm.glob('*.tif'))
    df_dsm = flist_to_df(flist_dsm)
    df_dsm['tile_id'] = df_dsm.apply(lambda x: x.row + '_' + x.col, axis=1)
    move_and_rename_processed_tiles(df_dsm, settings.SITE_NAME, settings.TARGET_DIR_DSM, 'DSM', move=False)
    logging.info('Finished moving and renaming DSM tiles!')

    # #### Create footprints file 

    TMP_MASK_VECTORIZE_DIR = PRODUCT_DIR / 'tmp_footprints'  # Path(r'D:\Pix4D_Processing\test')
    os.makedirs(TMP_MASK_VECTORIZE_DIR, exist_ok=True)
    Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'
    FOOTPRINTS_FILE = PRODUCT_DIR / f'{settings.SITE_NAME}_tile_footprints.geojson'

    logging.info(f'Start merging footprints to file {FOOTPRINTS_FILE}!')
    # create vector mask of Data (DN=0 for noData, DN=255 for valid Data)
    flist_out = list(settings.TARGET_DIR_ORTHO.glob('*.tif'))
    vector_list = Parallel(n_jobs=40)(
        delayed(create_mask_vector)(infile, TMP_MASK_VECTORIZE_DIR) for infile in tqdm.tqdm_notebook(flist_out[:]))

    # Merge vectors and remove noData parts
    gdf_list = Parallel(n_jobs=40)(
        delayed(load_and_prepare_footprints)(vector_file) for vector_file in tqdm.tqdm_notebook(vector_list[:]))

    merge_single_vector_files(gdf_list, FOOTPRINTS_FILE, settings.SITE_NAME, date)
    logging.info('Finished processing!')

    # remove NoData
    logging.info('Deleting empty tiles!')
    delete_empty_product_tiles(FOOTPRINTS_FILE, settings.TARGET_DIR_ORTHO, settings.TARGET_DIR_DSM)

    # Cleanup temporary dir
    shutil.rmtree(TMP_MASK_VECTORIZE_DIR)

    logging.info('Start postprocessing DSM tiles!')
    # Clip DSM to footprints
    df = gpd.read_file(FOOTPRINTS_FILE)
    fnames = df['DSM']
    DSM_DIR_TMP = settings.TARGET_DIR_DSM.parent / 'DSM_tmp'
    delete_input = True
    os.makedirs(DSM_DIR_TMP)
    Parallel(n_jobs=40)(
        delayed(clip_dsm_to_bounds)(FOOTPRINTS_FILE, filename, settings.TARGET_DIR_DSM, DSM_DIR_TMP) for filename in
        tqdm.tqdm(fnames[:]))
    if delete_input:
        shutil.rmtree(settings.TARGET_DIR_DSM)
        os.rename(DSM_DIR_TMP, settings.TARGET_DIR_DSM)

    logging.info('Finished postprocessing DSM tiles!')

    logging.info('Calculating Ortho Pyramids!')
    flist_ortho = list(settings.TARGET_DIR_ORTHO.glob('*.tif'))
    _ = Parallel(n_jobs=40)(delayed(calculate_pyramids)(filename) for filename in tqdm.tqdm_notebook(flist_ortho[:]))

    logging.info('Calculating DSM Pyramids!')
    flist_dsm = list(settings.TARGET_DIR_DSM.glob('*.tif'))
    _ = Parallel(n_jobs=40)(delayed(calculate_pyramids)(filename) for filename in tqdm.tqdm_notebook(flist_dsm[:]))

    # Create VRT files
    working_dir = Path(os.getcwd())
    vrt_path = working_dir / 'create_vrt.py'
    create_vrt(products_dir=PRODUCT_DIR, vrt_script_location=vrt_path)
    os.chdir(working_dir)

    # create previews
    create_previews(products_dir=PRODUCT_DIR, pyramid_level=1, overwrite=True)

    # Copy processing report, nav file log file
    logging.info('Copying reports!')
    processing_info_dir = PRODUCT_DIR / 'processing_info'
    os.makedirs(processing_info_dir, exist_ok=True)

    report_file = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.SITE_NAME / '1_initial' / 'report' / f'{settings.PIX4d_PROJECT_NAME}_report.pdf '
    shutil.copy(report_file, processing_info_dir)

    nav_file_in = Path(settings.PROJECT_DIR) / '01_rawdata' / 'tif' / 'geo_pix4d_new.txt'
    nav_file_out = processing_info_dir / f'{settings.SITE_NAME}_nav.txt'

    logging.info('Finished Postprocessing!')
    shutil.copy(nav_file_in, nav_file_out)
    shutil.copy(logfile, processing_info_dir)


if __name__ == "__main__":
    main()
