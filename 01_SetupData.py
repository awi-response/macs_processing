import geopandas as gpd
import shutil
import os
import glob
import pandas as pd
import sys
import numpy as np
import tqdm
import zipfile
from pathlib import Path
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import logging
from processing_utils import *
from utils_postprocessing import *

from MACS_00_Settings import *

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

###### START ###


pwd = Path(os.getcwd())

with zipfile.ZipFile(zippath, 'r') as zip_ref:
    zip_ref.extractall(PROJECT_DIR)
shutil.copy(nav_script_path, outdir)

# Copy AOI 
aoi_target = PROJECT_DIR / '02_studysites' / 'AOI.shp'
gpd.read_file(AOI).to_file(aoi_target)

# logger
logfile = PROJECT_DIR / f'{PROJECT_DIR.name}.log'
logging.basicConfig(#filename=PROJECT_DIR / f'{PROJECT_DIR.name}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(message)s', 
                    handlers=[logging.FileHandler(logfile),
                              logging.StreamHandler(sys.stdout)
                        ])
logging.info('Creation of logfile')
logging.info('Creating footprints selection')

for projects_file, parent_dir in zip(PROJECTS_FILES, PARENT_DIRS):
    dss = []
    print(parent_dir.parent)
    ds = get_overlapping_ds(AOI, projects_file, parent_dir)
    #dss.append(ds)
    if len(ds) > 0:
        break
stats = get_dataset_stats(ds, parent_dir, AOI)
print(stats)

# #### Select Dataset ID 
dataset_id = input('Please select ID: ')
footprints = retrieve_footprints(ds, dataset_id, parent_dir, AOI)
print("Total number of images:", len(footprints))
print("NIR images:", (footprints['Looking'] == 'center').sum())
print("RGB right images:", (footprints['Looking'] == 'right').sum())
print("RGB left images:", (footprints['Looking'] == 'left').sum())

footprints.to_file(path_footprints)
logging.info(f'Footprints file save to {path_footprints}')

# #### Load filtered footprints file 
dataset_name = get_dataset_name(ds, dataset_id)
path_infiles = Path(parent_dir) / dataset_name

df_final = prepare_df_for_mipps(path_footprints, path_infiles)
df_final['full_path'] = df_final.apply(lambda x: f'"{x.full_path}"', axis=1)

print("Total number of images:", len(df_final))
print("NIR images:", (df_final['Looking'] == 'center').sum())
print("RGB right images:", (df_final['Looking'] == 'right').sum())
print("RGB left images:", (df_final['Looking'] == 'left').sum())

# ### Run Process 
os.chdir(MIPPS_DIR)

max_roll = 3 # Select maximum roll angle to avoid image issues - SET in main settings part?
chunksize = 20 # this is a mipps-script thing

logging.info(f'Start exporting MACS files to TIFF using DLR mipps')
logging.info(f"Total number of images: {len(df_final)}")
logging.info(f"NIR images: {(df_final['Looking'] == 'center').sum()}")
logging.info(f"RGB right images: {(df_final['Looking'] == 'right').sum()}")
logging.info(f"RGB left images:{(df_final['Looking'] == 'left').sum()}")

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
    for df in tqdm.tqdm(np.array_split(df_nir, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_nir} -o={outdir} -j=4 {outlist}'
        os.system(s)

# this is RGB
if 'right' in sensors:
    logging.info(f'Start transforming RGB right files')
    logging.info(f'MIPPS Script: {mipps_script_right.name}')
    
    looking = 'right'
    q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
    df_right = df_final[q]
    split = len(df_right) // chunksize
    if split == 0: split+=1
    for df in tqdm.tqdm(np.array_split(df_right, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_right} -o={outdir} -j=4 {outlist}'
        os.system(s)

if 'left' in sensors:
    logging.info(f'Start transforming RGB left files')
    logging.info(f'MIPPS Script: {mipps_script_left.name}')
    
    looking = 'left'
    q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
    df_left = df_final[q]
    split = len(df_left) // chunksize
    if split == 0: split+=1
    for df in tqdm.tqdm(np.array_split(df_left, split)):
        outlist = ' '.join(df['full_path'].values[:])
        s = f'{MIPPS_BIN} -c={mipps_script_left} -o={outdir} -j=4 {outlist}'
        os.system(s)

# ### Rescale image values 

# #### Image Statistics 

if SCALING:
    logging.info(f'Start reading Image statistics')
    df_stats = get_image_stats_multi(OUTDIR, sensors, nth_images=1, quiet=False, n_jobs=40)
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
        
        _ = Parallel(n_jobs=n_jobs)(delayed(write_new_values)(image, scale_lower, scale_upper, shutter_factor=shutter_factor, tag=True) for image in tqdm.tqdm(images[:]))
    logging.info(f'Finished Image Scaling')

# #### Crop Corners of images 
if CROP_CORNER:
    logging.info(f'Start Cropping corners')
    logging.info(f'Disk Size: {DISK_SIZE}')
    #mask = make_mask((3232, 4864), disksize=DISK_SIZE)
    for sensor in sensors[:]:
        mask = make_mask((3232, 4864), disksize=DISK_SIZE)
        images = list(OUTDIR[sensor].glob('*'))
        if sensor != 'nir':
            mask = np.r_[[mask]*3]
        _ = Parallel(n_jobs=4)(delayed(mask_and_tag)(image, mask, tag=None) for image in tqdm.tqdm(images))
    logging.info(f'Finished Cropping corners')


# #### Write exif information into all images 

logging.info(f'Start writing EXIF Tags')
for sensor in tqdm.tqdm(sensors):
    print(sensor)
    write_exif(OUTDIR[sensor], tag[sensor], EXIF_PATH)
logging.info(f'Finished writing EXIF Tags')

# #### Nav
logging.info(f'Start preparing nav file')

navfile = list(Path(path_infiles).glob('*nav.txt'))[0]
shutil.copy(navfile, OUTDIR['nir'].parent / 'nav.txt')
os.chdir(OUTDIR['nir'].parent)
os.system('python pix4dnav.py')

logging.info(f'Finished preparing nav file')