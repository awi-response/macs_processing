import argparse
import importlib
import logging
import sys
# ignore warnings
import warnings

from src.macs_processing.utils.processing import *
from utils_postprocessing import *
from utils_wbt import *
import shutil

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", required=True, type=Path, help="Path to Settings file")

parser.add_argument("-dsm", "--dsm_mode", default='wbt', type=str, choices=['pix4d', 'wbt'],
                    help='Set dsm_processing_mode, "pix4d" for using pix4d provided dsm tiles; "wbt" for custom '
                         'whiteboxtools created DSM')

parser.add_argument("-pc", "--point_cloud", default='both', type=str, choices=["both", "nir", "rgb"],
                    help='Set which point cloud to use. Options: "both", "nir", "rgb"')

parser.add_argument("-m", "--mosaic", action='store_true',
                    help='Set flag to calculate COG mosaic')

parser.add_argument("-keep_dsm_if", "--keep_dsm_if", action='store_true',
                    help="keep intermediate DSM files (for debugging)")

parser.add_argument("--n_jobs", default=40, type=int,
                    help='Set number of parallel processes')


args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###

PROCESS = True

def main():
    # Setup processing paths - make more consistent
    PRODUCT_DIR = Path(settings.PROJECT_DIR) / '06_DataProducts'
    settings.TARGET_DIR_ORTHO = PRODUCT_DIR / 'Ortho'
    # temporary processing dirs
    TMP_DIR_ORTHO = PRODUCT_DIR / 'tmp_mosaics'
    TMP_DIR_VRT = PRODUCT_DIR / 'tmp_vrt'
    TMP_MASK_VECTORIZE_DIR = PRODUCT_DIR / 'tmp_footprints'
    settings.TARGET_DIR_ORTHO = PRODUCT_DIR / 'Ortho'
    settings.TARGET_DIR_DSM = PRODUCT_DIR / 'DSM'
    settings.TARGET_DIR_PC = PRODUCT_DIR / 'PointClouds'

    tiles_dir = Path(
        settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
    
    # get region properties, date and resolution
    region, site, site_number, date, resolution = parse_site_name_v2(settings.SITE_NAME)
    #create target/output dirs
    os.makedirs(settings.TARGET_DIR_ORTHO, exist_ok=True)
    os.makedirs(settings.TARGET_DIR_DSM, exist_ok=True)


    # get tile_ids
    flist = list(tiles_dir.glob('*.tif'))
    df = flist_to_df(flist)
    df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)
    tiles = pd.unique(df['tile_id'])

        # Get sensor names
    nir_sensor = get_nir_sensor_name(df)
    rgb_sensor = get_rgb_sensor_name(df)

    # Validate input files - raise error if file sets are completely empty
    content_is_valid = check_ortho_validity(df)
    if not content_is_valid:
        raise Exception('Processed Orthotiles do not contain Data')

    #### Run
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(
            delayed(full_postprocessing_optical)(df, tile, nir_name=nir_sensor, rgb_name=rgb_sensor, vrt_dir=TMP_DIR_VRT, target_dir_mosaic=TMP_DIR_ORTHO) for tile in tqdm.tqdm(tiles[:]))
        logging.info('Finished postprocessing Orthoimage tiles!')

    # #### Move and rename to output

    if True:
        flist = list(TMP_DIR_ORTHO.glob('mosaic*.tif'))
        df = flist_to_df(flist)
        df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)

        # TODO: uncomment
        move_and_rename_processed_tiles(df, settings.SITE_NAME, settings.TARGET_DIR_ORTHO, 'Ortho', move=True)
        logging.info('Finished moving and renaming Ortho tiles!')

    # #### Create footprints file
    os.makedirs(TMP_MASK_VECTORIZE_DIR, exist_ok=True)
    # TODO: This is doing nothing
    Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'
    
    # make geopackage - perhaps in the end
    FOOTPRINTS_FILE = PRODUCT_DIR / f'{settings.SITE_NAME}_tile_footprints.geojson'

    logging.info(f'Start merging footprints to file {FOOTPRINTS_FILE}!')
    # create vector mask of Data (DN=0 for noData, DN=255 for valid Data)
    flist_out = list(settings.TARGET_DIR_ORTHO.glob('*.tif'))
    
    if True:
        vector_list = Parallel(n_jobs=args.n_jobs)(
            delayed(create_mask_vector)(infile, TMP_MASK_VECTORIZE_DIR) for infile in tqdm.tqdm(flist_out[:]))
    
    # Merge vectors and remove noData parts
    if True:
        gdf_list = Parallel(n_jobs=args.n_jobs)(
            delayed(load_and_prepare_footprints)(vector_file) for vector_file in tqdm.tqdm(vector_list[:]))
    
    merge_single_vector_files(gdf_list, FOOTPRINTS_FILE, settings.SITE_NAME, date)
    
    logging.info('Finished processing!')
    # remove NoData
    logging.info('Deleting empty tiles!')
    if True:
        delete_empty_product_tiles(FOOTPRINTS_FILE, settings.TARGET_DIR_ORTHO, settings.TARGET_DIR_DSM)


   # ############### CLIP Point Clouds to footprint ################### #
    # extract individual tiles from footprints file
    logging.info('Start tiling Point Clouds!')
    os.makedirs(settings.TARGET_DIR_PC, exist_ok=True)

    # point cloud
    point_clouds_dir = settings.Path(
                settings.PROJECT_DIR) / '04_pix4d' / settings.PIX4d_PROJECT_NAME / '2_densification' / 'point_cloud'
    point_cloud_nir = list(point_clouds_dir.glob(f'*{nir_sensor}_densified_point_cloud.las'))[0]
    point_cloud_rgb = list(point_clouds_dir.glob(f'*{rgb_sensor}_densified_point_cloud.las'))[0]
    # RUN Point Cloud Clipping
    # NIR Point Cloud
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(delayed(create_point_cloud_tiles_las2las)
                               (point_cloud_nir,
                                tile, settings,
                                target_dir=settings.TARGET_DIR_PC,
                                product_name='PointCloudNIR')
                               for tile in tqdm.tqdm(vector_list[:]))
    # RGB Point Cloud
    if True:
        _ = Parallel(n_jobs=args.n_jobs)(delayed(create_point_cloud_tiles_las2las)
                               (point_cloud_rgb,
                                tile, settings,
                                target_dir=settings.TARGET_DIR_PC,
                                product_name='PointCloudRGB')
                               for tile in tqdm.tqdm(vector_list[:]))

    logging.info('Finished tiling Point Clouds!')

    copy_tgt = Path(r'S:\p_macsprocessing\data_products\v2') / settings.PIX4d_PROJECT_NAME / 'PointClouds'
    print('Copy files to', copy_tgt)
    shutil.copytree(settings.TARGET_DIR_PC, copy_tgt, dirs_exist_ok=True)

    

if __name__ == "__main__":
    main()
