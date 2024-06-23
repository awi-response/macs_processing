
from pathlib import Path
import pandas as pd
import os
pwd = Path(os.getcwd())
print(pwd)

##### PREPROCESSING SETTINGS ############

# Set project directory here, where you want to process your data
SITE_NAME = Path(__file__).stem#'WC_TrailValleyCreek_20180822_10cm_02' # e.g. NWC_TukRoad_05_20180822_10cm
PROJECT_DIR = Path(r'E:\MACS_Batch_Processing\Batch2024_v2') / SITE_NAME

# SET AOI for footprints selection
#AOI = Path(r'test_data\2023_driftwood_Test01.gpkg')
AOI = Path(r'S:\p_macsprocessing\aoi_2023') / f'{SITE_NAME}.gpkg' # Must be a valid link to a polygon vector file

######## Everything below can be kept

PIX4d_PROJECT_NAME = SITE_NAME

# TODO: change to NIR, RGB
# determine which sensors to include in processing (possible options: 'left', 'right', 'nir')
sensors = ['left', 'right', 'nir']

# SET SCALING 
SCALING = 0
SCALE_LOW = False # Set to True to use calculated lower boundary - skews NDVI
SCALE_HIGH = False # Set to True to use calculated upper boundary
SCALING_EMPIRICAL = False

# Set CROP CORNER if 
CROP_CORNER = 0 # SET to 1 if you want to crop corners (set to NoData)
DISK_SIZE = 5200 # Cropping diameter, the larger the fewer no data


# ### Imports
#pwd = Path(os.getcwd())

PARENT_DIRS = [
              Path(r'N:\Response\Restricted_Airborne\MACS\Alaska\2019_ThawTrend-Air\raw_data'),
              Path(r'N:\response\Restricted_Airborne\MACS\Alaska\2021_Perma-X_Alaska\01_raw_data'),
              Path(r'N:\Response\Restricted_Airborne\MACS\Canada\2023_Perma-X_Canada\1_MACS_original_images')
              ]
              
# imageing projects footprint file
PROJECTS_FILES = [                  Path(r'N:\Response\Restricted_Airborne\MACS\Alaska\2019_ThawTrend-Air\footprints\ThawTrendAir2019_MACS_footprints_datasets.shp'),
                  Path(r'N:\Response\Restricted_Airborne\MACS\Alaska\2021_Perma-X_Alaska\03_footprints\Perma-X\PermaX2021_MACS_footprints_datasets.shp'),
                  Path(r'N:\Response\Restricted_Airborne\MACS\Canada\2023_Perma-X_Canada\03_footprints\PermaX2023_MACS_footprints_datasets.gpkg')
                 ]

CODE_DIR = pwd
MIPPS_DIR = r'C:\Program Files\DLR MACS-Box\bin'
MIPPS_BIN = r'..\tools\MACS\mipps.exe'
EXIF_PATH = Path(CODE_DIR / Path(r'exiftool\exiftool.exe'))
mipps_script_dir = Path('mipps_scripts')

# Define the index values
index_values = ['33552', '33576', '33577', 'RGB 121502', 'NIR 99683']

# Create an empty DataFrame with the specified index values
df = pd.DataFrame(index=index_values)

mipps_script_nir = '33552_all_taps_2018-09-26_12-58-15_modelbased.mipps'
mipps_script_right = '33576_all_taps_2018-09-26_13-13-43_modelbased.mipps'
mipps_script_left = '33577_all_taps_2018-09-26_13-21-24_modelbased.mipps'

mipps_script_nir = pwd / mipps_script_dir / mipps_script_nir
mipps_script_right = pwd / mipps_script_dir / mipps_script_right
mipps_script_left = pwd / mipps_script_dir / mipps_script_left

DATA_DIR = PROJECT_DIR / '01_rawdata' / 'tif'
OUTDIR = {'right': DATA_DIR / Path('33576_Right'),
          'left':DATA_DIR / Path('33577_Left'),
          'nir':DATA_DIR / Path('33552_NIR')}
tag = {'right':'MACS_RGB_Right_33576',
       'left':'MACS_RGB_Left_33577',
       'nir':'MACS_NIR_33552'}
"""
# find out sensor id
tag = {'right':'33576',
       'left':'33577',
       'nir':'33552'}
"""

# Path of filtered footprints file (.shp file)
path_footprints = PROJECT_DIR / '02_studysites' / 'footprints.gpkg'
outdir = PROJECT_DIR / '01_rawdata' / 'tif'


# #### Prepare processing dir 
# * check if exists
zippath = CODE_DIR / 'processing_folder_structure_template.zip'
nav_script_path = CODE_DIR / 'pix4dnav.py'