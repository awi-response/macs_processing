import argparse
import sys
import logging
from processing_utils import *
from utils_postprocessing import *
import importlib
import rasterio

# ignore warnings
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", default=Path('MACS_00_Settings.py'),
                    type=Path,
                    help="Path to Settings file")
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

    # Parallel version crashes
    _ = Parallel(n_jobs=40)(
        delayed(full_postprocessing_optical)(df, tile, nir_name=nir_sensor) for tile in tqdm.tqdm_notebook(tiles[:]))

    logging.info('Finished postprocessing Orthoimage tiles!')

    # #### PostProcess DSM
    logging.info('Start postprocessing DSM tiles!')

    tiles_dir_dsm = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles'
    flist_dsm = list(tiles_dir_dsm.glob('*.tif'))

    _ = Parallel(n_jobs=40)(delayed(calculate_pyramids)(filename) for filename in tqdm.tqdm_notebook(flist_dsm[:]))

    logging.info('Finished postprocessing DSM tiles!')

    # #### Rename

    PRODUCT_DIR = Path(settings.PROJECT_DIR) / '06_DataProducts'
    settings.TARGET_DIR_ORTHO = PRODUCT_DIR / 'Ortho'
    settings.TARGET_DIR_DSM = PRODUCT_DIR / 'DSM'

    region, site, site_number, date, resolution = parse_site_name(settings.SITE_NAME)
    os.makedirs(settings.TARGET_DIR_ORTHO, exist_ok=True)
    os.makedirs(settings.TARGET_DIR_DSM, exist_ok=True)

    # #### Create output dirs 

    # #### Move and rename to output 

    logging.info('Start moving and renaming Ortho tiles!')

    tiles_dir = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
    flist = list(tiles_dir.glob('mosaic*.tif'))
    df = flist_to_df(flist)
    df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)
    # tiles = pd.unique(df['tile_id'])

    move_and_rename_processed_tiles(df, settings.SITE_NAME, settings.TARGET_DIR_ORTHO, 'Ortho', move=False)
    logging.info('Finished moving and renaming Ortho tiles!')

    # #### DSM 

    logging.info('Start moving and renaming DSM tiles!')

    tiles_dir_dsm = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles'
    flist_dsm = list(tiles_dir_dsm.glob('*.tif'))
    df_dsm = flist_to_df(flist_dsm)
    df_dsm['tile_id'] = df_dsm.apply(lambda x: x.row + '_' + x.col, axis=1)

    move_and_rename_processed_tiles(df_dsm, settings.SITE_NAME, settings.TARGET_DIR_DSM, 'DSM', move=False)

    logging.info('Finished moving and renaming DSM tiles!')

    # #### Create footprints file 

    TMP_MASK_VECTORIZE_DIR = PRODUCT_DIR / 'tmp_footprints'  # Path(r'D:\Pix4D_Processing\test')
    os.makedirs(TMP_MASK_VECTORIZE_DIR, exist_ok=True)
    GDAL_POLYGONIZE = Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'
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
