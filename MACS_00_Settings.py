
from pathlib import Path

##### PEPROCESSING SETTINGS ############

# Set project directory here, where you want to process your data
SITE_NAME = 'NWC_TukRoad_03_20180822_10cm'

PROJECT_DIR = Path(r'D:\Pix4D_Processing\tests') / SITE_NAME

# SET AOI for footprints selection
AOI = Path(r'S:\p_macsprocessing\test_sites\site_definitions') / 'test_TUK.shp'

PIX4d_PROJECT_NAME = SITE_NAME

# determine which sensors to include in processing (possible options: 'left', 'right', 'nir')
sensors = ['left', 'right', 'nir']

# SET SCALING 
SCALING = 1
SCALE_LOW = False # Set to True to use calculated lower boundary - skews NDVI
SCALE_HIGH = True # Set to True to use calculated upper boundary
SCALING_EMPIRICAL = False


# Set CROP CORNER if 
CROP_CORNER = 0 # SET to 1 if you want to crop corners (set to NoData)
DISK_SIZE = 5200 # Cropping diameter, the larger the fewer no data


# ### Imports 

import geopandas as gpd
import shutil
import os
import glob
import pandas as pd
from IPython.display import clear_output
import sys
import numpy as np
import tqdm
import zipfile
from pathlib import Path
from joblib import delayed, Parallel, wrap_non_picklable_objects
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from processing_utils import *
from utils_postprocessing import *

pwd = Path(os.getcwd())


# 2018 Canada
PARENT_DIRS = [Path(r'N:\response\Restricted_Airborne\MACs\Canada\1_MACS_original_images'),
              Path(r'N:\response\Restricted_Airborne\MACs\Alaska\ThawTrend-Air_2019\raw_data'),
              Path(r'N:\response\Restricted_Airborne\MACs\2021_Perma-X_Alaska\01_raw_data')
              ]
              
# imageing projects footprint file
PROJECTS_FILES = [Path(r'N:\response\Restricted_Airborne\MACs\Canada\0_MACS_important_files\1_all_canada_flights_2018_footprints\all_canada_flights_2018_footprints_datasets.shp'),
                  Path(r'N:\response\Restricted_Airborne\MACs\Alaska\ThawTrend-Air_2019\footprints\ThawTrendAir2019_MACS_footprints_datasets.shp'),
                  Path(r'N:\response\Restricted_Airborne\MACs\2021_Perma-X_Alaska\03_footprints\Perma-X\PermaX2021_MACS_footprints_datasets.shp')
                 ]

CODE_DIR = pwd
MIPPS_DIR = r'C:\Program Files\DLR MACS-Box\bin'
MIPPS_BIN = r'..\tools\MACS\mipps.exe'
EXIF_PATH = Path(CODE_DIR / Path(r'exiftool\exiftool.exe'))
mipps_script_dir = Path('mipps_scripts')

mipps_script_nir = '33552_all_taps_2018-09-26_12-58-15_modelbased.mipps'
mipps_script_right = '33576_all_taps_2018-09-26_13-13-43_modelbased.mipps'
mipps_script_left = '33577_all_taps_2018-09-26_13-21-24_modelbased.mipps'

mipps_script_nir = pwd / mipps_script_dir / mipps_script_nir
mipps_script_right = pwd / mipps_script_dir / mipps_script_right
mipps_script_left = pwd / mipps_script_dir / mipps_script_left

DATA_DIR = Path(PROJECT_DIR) / '01_rawdata' / 'tif'
OUTDIR = {'right': DATA_DIR / Path('33576_Right'),
          'left':DATA_DIR / Path('33577_Left'),
          'nir':DATA_DIR / Path('33552_NIR')}
tag = {'right':'MACS_RGB_Right_33576',
       'left':'MACS_RGB_Left_33577',
       'nir':'MACS_NIR_33552'}

# Path of filtered footprints file (.shp file)
path_footprints = Path(PROJECT_DIR) / '02_studysites' / 'footprints.shp'
outdir = os.path.join(PROJECT_DIR, '01_rawdata','tif')


# #### Prepare processing dir 
# * check if exists

# In[ ]:


zippath = os.path.join(CODE_DIR, 'processing_folder_structure_template.zip')
nav_script_path = os.path.join(CODE_DIR, 'pix4dnav.py')


# In[ ]:
"""

with zipfile.ZipFile(zippath, 'r') as zip_ref:
    zip_ref.extractall(PROJECT_DIR)
shutil.copy(nav_script_path, outdir)


# In[ ]:


# Copy AOI 
aoi_target = PROJECT_DIR / '02_studysites' / 'AOI.shp'
gpd.read_file(AOI).to_file(aoi_target)


# In[ ]:


# logger
logging.basicConfig(filename=PROJECT_DIR / f'{PROJECT_DIR.name}.log', level=logging.INFO, format='%(asctime)s %(message)s', )
logging.info('Creation of logfile')


# ### Auto Selection of footprints
# provide more info

# In[ ]:


logging.info('Creating footprints selection')


# In[ ]:


for projects_file, parent_dir in zip(PROJECTS_FILES, PARENT_DIRS):
    dss = []
    print(parent_dir.parent)
    ds = get_overlapping_ds(AOI, projects_file, parent_dir)
    #dss.append(ds)
    if len(ds) > 0:
        break
stats = get_dataset_stats(ds, parent_dir, AOI)
stats


# #### Select Dataset ID 

# In[ ]:


dataset_id = input('Please select ID: ')
footprints = retrieve_footprints(ds, dataset_id, parent_dir, AOI)
print("Total number of images:", len(footprints))
print("NIR images:", (footprints['Looking'] == 'center').sum())
print("RGB right images:", (footprints['Looking'] == 'right').sum())
print("RGB left images:", (footprints['Looking'] == 'left').sum())


# In[ ]:


footprints.to_file(path_footprints)
logging.info(f'Footprints file save to {path_footprints}')


# #### Load filtered footprints file 

# In[ ]:


dataset_name = get_dataset_name(ds, dataset_id)
path_infiles = Path(parent_dir) / dataset_name


# In[ ]:


df_final = prepare_df_for_mipps(path_footprints, path_infiles)
df_final['full_path'] = df_final.apply(lambda x: f'"{x.full_path}"', axis=1)


# In[ ]:


print("Total number of images:", len(df_final))
print("NIR images:", (df_final['Looking'] == 'center').sum())
print("RGB right images:", (df_final['Looking'] == 'right').sum())
print("RGB left images:", (df_final['Looking'] == 'left').sum())


# ### Run Process 

# In[ ]:


os.chdir(MIPPS_DIR)


# In[ ]:


max_roll = 3 # Select maximum roll angle to avoid image issues - SET in main settings part?
chunksize = 20 # this is a mipps-script thing


# #### Export MACS to TIFF

# In[ ]:


logging.info(f'Start exporting MACS files to TIFF using DLR mipps')
logging.info(f"Total number of images: {len(df_final)}")
logging.info(f"NIR images: {(df_final['Looking'] == 'center').sum()}")
logging.info(f"RGB right images: {(df_final['Looking'] == 'right').sum()}")
logging.info(f"RGB left images:{(df_final['Looking'] == 'left').sum()}")


# In[ ]:


# this is relevant for NIR only
if 'nir' in sensors:
    logging.info(f'Start transforming NIR files')
    logging.info(f'MIPPS Script: {mipps_script_nir.name}')
    
    looking = 'center'
    q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
    df_nir = df_final[q]
    print(len(df_nir))
    split = len(df_nir) // chunksize
    if split == 0: split+=1
    for df in tqdm.tqdm_notebook(np.array_split(df_nir, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_nir} -o={outdir} -j=4 {outlist}'
        os.system(s)
        #print(s)


# In[ ]:


# this is RGB
if 'right' in sensors:
    logging.info(f'Start transforming RGB right files')
    logging.info(f'MIPPS Script: {mipps_script_right.name}')
    
    looking = 'right'
    q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
    df_right = df_final[q]
    split = len(df_right) // chunksize
    if split == 0: split+=1
    for df in tqdm.tqdm_notebook(np.array_split(df_right, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_right} -o={outdir} -j=4 {outlist}'
        os.system(s)


# In[ ]:


if 'left' in sensors:
    logging.info(f'Start transforming RGB left files')
    logging.info(f'MIPPS Script: {mipps_script_left.name}')
    
    looking = 'left'
    q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
    df_left = df_final[q]
    split = len(df_left) // chunksize
    if split == 0: split+=1
    for df in tqdm.tqdm_notebook(np.array_split(df_left, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_left} -o={outdir} -j=4 {outlist}'
        os.system(s)


# ### Rescale image values 

# #### Image Statistics 

# In[ ]:


if SCALING:
    logging.info(f'Start reading Image statistics')
    get_ipython().run_line_magic('time', 'df_stats = get_image_stats_multi(OUTDIR, sensors, nth_images=1, quiet=False, n_jobs=40)')
    #absolute
    if SCALE_LOW:
        scale_lower = int(df_stats['min'].mean().round())
    else:
        scale_lower = 1
    if SCALE_HIGH:
        scale_upper = int(df_stats['max'].mean().round())
    else:
        scale_upper = 2*16-1
    print(f'Mean of minimums: {scale_lower}')
    print(f'Mean of maximums: {scale_upper}')
    logging.info(f'Mean of minimums: {scale_lower}')
    logging.info(f'Mean of maximums: {scale_upper}')
    logging.info(f'Finished reading Image statistics')


# In[ ]:


if SCALING_EMPIRICAL:
    # empirical
    df_stats2 = df_stats.replace({'left':'RGB', 'right':'RGB'})
    grouped = df_stats2.groupby('sensor').mean()
    empirical_scale_factor = (grouped.loc['nir'] / grouped.loc['RGB'] / 0.8)['min']
else: 
    empirical_scale_factor = 1


# #### Run scaling
# * minimum default to 1
# * consistency for final index calculation

# In[ ]:


if SCALING:
    logging.info(f'Start Image Scaling')
    n_jobs = 20
    for sensor in sensors:
        print(f'Processing {sensor}')
        #shutter_factor
        images = list(OUTDIR[sensor].glob('*.tif'))[:]
        if sensor in ['right', 'left']:
            shutter_factor = get_shutter_factor(OUTDIR, sensors) * empirical_scale_factor
            print(f'RGB to NIR factor = {shutter_factor}')
        else:
            shutter_factor = 1
        
        get_ipython().run_line_magic('time', '_ = Parallel(n_jobs=n_jobs)(delayed(write_new_values)(image, scale_lower, scale_upper, shutter_factor=shutter_factor, tag=True) for image in tqdm.tqdm_notebook(images[:]))')
    logging.info(f'Finished Image Scaling')


# #### Crop Corners of images 

# In[ ]:


if CROP_CORNER:
    logging.info(f'Start Cropping corners')
    logging.info(f'Disk Size: {DISK_SIZE}')
    #mask = make_mask((3232, 4864), disksize=DISK_SIZE)
    for sensor in sensors[:]:
        mask = make_mask((3232, 4864), disksize=DISK_SIZE)
        images = list(OUTDIR[sensor].glob('*'))
        if sensor != 'nir':
            mask = np.r_[[mask]*3]
        get_ipython().run_line_magic('time', '_ = Parallel(n_jobs=4)(delayed(mask_and_tag)(image, mask, tag=None) for image in tqdm.tqdm_notebook(images))')
    logging.info(f'Finished Cropping corners')


# #### Write exif information into all images 

# In[ ]:


logging.info(f'Start writing EXIF Tags')
for sensor in tqdm.tqdm_notebook(sensors):
    print(sensor)
    get_ipython().run_line_magic('time', 'write_exif(OUTDIR[sensor], tag[sensor], EXIF_PATH)')
logging.info(f'Finished writing EXIF Tags')


# #### Nav

# In[ ]:


logging.info(f'Start preparing nav file')

navfile = list(Path(path_infiles).glob('*nav.txt'))[0]
shutil.copy(navfile, OUTDIR['nir'].parent / 'nav.txt')
os.chdir(OUTDIR['nir'].parent)
os.system('python pix4dnav.py')

logging.info(f'Finished preparing nav file')


# # Now Run Pix4d Processing 

# ## Post Processing 

# ## Mosaic 

# ### Tiling and renaming
# * check pix4d tiling system
#   * using same system
# * create tiling check for pyramids

# Todo:
# 3. rename
# 

# #### PostProcess Optical
# * create 4 band mosaic + calculate pyramids

# In[ ]:


logging.info(f'Start postprocessing Orthoimage tiles!')

tiles_dir = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
flist = list(tiles_dir.glob('*.tif'))
df = flist_to_df(flist)
df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)
tiles = pd.unique(df['tile_id'])

#### Run 

# Parallel version crashes
get_ipython().run_line_magic('time', '_ = Parallel(n_jobs=40)(delayed(full_postprocessing_optical)(df, tile) for tile in tqdm.tqdm_notebook(tiles[:]))')

logging.info(f'Finished postprocessing Orthoimage tiles!')


# #### PostProcess DSM

# In[ ]:


logging.info(f'Start postprocessing DSM tiles!')

tiles_dir_dsm = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles'
flist_dsm = list(tiles_dir_dsm.glob('*.tif'))

get_ipython().run_line_magic('time', '_ = Parallel(n_jobs=40)(delayed(calculate_pyramids)(filename) for filename in tqdm.tqdm_notebook(flist_dsm[:]))')

logging.info(f'Finished postprocessing DSM tiles!')


# #### Rename
# Move to settings

# WA_KobukDelta_02_20210710_20cm

# In[ ]:


PRODUCT_DIR = Path(PROJECT_DIR) / '06_DataProducts'
TARGET_DIR_ORTHO = tiles_dir = PRODUCT_DIR / 'Ortho'
TARGET_DIR_DSM = tiles_dir = PRODUCT_DIR / 'DSM'

region, site, site_number, date, resolution = parse_site_name(SITE_NAME)
os.makedirs(TARGET_DIR_ORTHO, exist_ok=True)
os.makedirs(TARGET_DIR_DSM, exist_ok=True)


# #### Create output dirs 

# #### Move and rename to output 

# In[ ]:


logging.info(f'Start moving and renaming Ortho tiles!')

tiles_dir = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic' / 'tiles'
flist = list(tiles_dir.glob('mosaic*.tif'))
df = flist_to_df(flist)
df['tile_id'] = df.apply(lambda x: x.row + '_' + x.col, axis=1)
tiles = pd.unique(df['tile_id'])

move_and_rename_processed_tiles(df, SITE_NAME, TARGET_DIR_ORTHO, 'Ortho', move=False)
logging.info(f'Finished moving and renaming Ortho tiles!')


# #### DSM 

# In[ ]:


logging.info(f'Start moving and renaming DSM tiles!')

tiles_dir_dsm = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm' / 'tiles'
flist_dsm = list(tiles_dir_dsm.glob('*.tif'))
df_dsm = flist_to_df(flist_dsm)
df_dsm['tile_id'] = df_dsm.apply(lambda x: x.row + '_' + x.col, axis=1)
tiles = pd.unique(df_dsm['tile_id'])

move_and_rename_processed_tiles(df_dsm, SITE_NAME, TARGET_DIR_DSM, 'DSM', move=False)

logging.info(f'Finished moving and renaming DSM tiles!')


# #### Create footprints file 

# In[ ]:


TMP_MASK_VECTORIZE_DIR = PRODUCT_DIR / 'tmp_footprints'#Path(r'D:\Pix4D_Processing\test')
os.makedirs(TMP_MASK_VECTORIZE_DIR, exist_ok=True)
GDAL_POLYGONIZE = Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'
FOOTPRINTS_FILE = PRODUCT_DIR / f'{SITE_NAME}_tile_footprints.geojson'

logging.info(f'Start merging footprints to file {FOOTPRINTS_FILE}!')
# create vector mask of Data (DN=0 for noData, DN=255 for valid Data)
flist_out = list(TARGET_DIR_ORTHO.glob('*.tif'))
vector_list = Parallel(n_jobs=40)(delayed(create_mask_vector)(infile, TMP_MASK_VECTORIZE_DIR) for infile in tqdm.tqdm_notebook(flist_out[:]))

# Merge vectors and remove noData parts
gdf_list = Parallel(n_jobs=40)(delayed(load_and_prepare_footprints)(vector_file) for vector_file in tqdm.tqdm_notebook(vector_list[:]))

merge_single_vector_files(gdf_list, FOOTPRINTS_FILE, SITE_NAME, date)
logging.info(f'Finished processing!')


# In[ ]:


# Cleanup temporary dir
shutil.rmtree(TMP_MASK_VECTORIZE_DIR)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Only full set 

# ### Create 4 band CIR mosaic

# Run this part after Processing in Pix4D

# In[ ]:


dir_processed = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '2_mosaic'
os.chdir(dir_processed)


# In[ ]:


rgbfile = list(Path('.').glob('*group1.tif'))[0]
nirfile = list(Path('.').glob('*nir.tif'))[0]
outmosaic = '_'.join(rgbfile.name.split('_')[:-3])+'_mosaic.tif'

dsm_dir = dir_processed = Path(PROJECT_DIR) / '04_pix4d' / PIX4d_PROJECT_NAME / '3_dsm_ortho' / '1_dsm'
dsm_file = list(dsm_dir.glob('*dsm.tif'))[0]


# #### Export single bands and merge afterwards 
# * make function

# In[ ]:


logging.info(f'Start restructuring and mosaicking final Orthomosaics!')

remove_files = False

for band in [1,2,3]:
    s = f'gdalbuildvrt -b {band} rgb_{band}.vrt {rgbfile}'
    os.system(s)

for band in [1]:
    s = f'gdalbuildvrt -b {band} nir_{band}.vrt {nirfile}'
    os.system(s)

s = f'gdalbuildvrt -separate 4band.vrt rgb_3.vrt rgb_2.vrt rgb_1.vrt nir_1.vrt'
os.system(s)

s = f'gdal_translate -a_nodata 0 -co COMPRESS=DEFLATE -co BIGTIFF=YES 4band.vrt {outmosaic}'
os.system(s)
if remove_files:
    for file in ['rgb_3.vrt', 'rgb_2.vrt', 'rgb_1.vrt', 'nir_1.vrt', '4band.vrt']:
        os.remove(file)
logging.info(f'Finished restructuring and mosaicking final Orthomosaics!')


# ### Mask 

# In[ ]:


logging.info(f'Start updating mask of Orthomosiacs!')
with rasterio.open(outmosaic, 'r+') as src:
    src.profile['nodata'] = 0
    data = src.read()
    newmask = ~(data == 0).any(axis=0)
    newmask_write = np.r_[src.count * [newmask]]
    data_masked = data * newmask_write
    src.set_band_description(1, 'MACS Blue Band')
    src.set_band_description(2, 'MACS Green Band')
    src.set_band_description(3, 'MACS Red Band')
    src.set_band_description(4, 'MACS NIR Band')
    src.write(data_masked)
logging.info(f'Finished updating mask of Orthomosiacs!')


# #### Calculate Pyramids 

# Ortho

# In[ ]:


logging.info(f'Start calculating pyramid layers for Orthoimage!!')
addo = f'gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE --config GDAL_NUM_THREADS ALL_CPUS {outmosaic}'
os.system(addo)
logging.info(f'Finished calculating pyramid layers for Orthoimage!')


# DSM

# In[ ]:


logging.info(f'Start calculating pyramid layers for DSM!')
addo = f'gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE --config GDAL_NUM_THREADS ALL_CPUS {dsm_file}'
os.system(addo)
logging.info(f'Finished calculating pyramid layers for DSM!')
"""
