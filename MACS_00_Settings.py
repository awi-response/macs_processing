
from pathlib import Path
import os

##### PREPROCESSING SETTINGS ############

# Set project directory here, where you want to process your data
SITE_NAME = ' ' # e.g. NWC_TukRoad_05_20180822_10cm
PROJECT_DIR = Path(r'D:\Pix4D_Processing\tests') / SITE_NAME

# SET AOI for footprints selection
AOI = Path(r' XXX.shp')
#AOI = Path(r'S:\p_macsprocessing\test_sites\site_definitions') / 'test_TUK.shp' # Must be a valid link to a polygon vector file

######## Everything below can be kept

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

zippath = os.path.join(CODE_DIR, 'processing_folder_structure_template.zip')
nav_script_path = os.path.join(CODE_DIR, 'pix4dnav.py')