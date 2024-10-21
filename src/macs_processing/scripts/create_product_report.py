#!/usr/bin/env python
# coding: utf-8

# # Create Product report for MACS processing data
# 
# ## 1. Data loading
# 
# ## 2. File checks
# * check completeness of input dirs and base files
# * count tiles
# * control for

# ## Imports

# In[1]:


import pandas as pd
from pathlib import Path
import itertools
import shutil
import geopandas as gpd
from tqdm.notebook import tqdm

from utils_report import *


# In[2]:


def check_vrt_is_linux(row):
    return all([vrt_is_linux(row['products_dir'] / item) for item in ['Ortho.vrt', 'DSM.vrt']])

def vrt_is_linux(vrt_file):
    with open(vrt_file, 'r') as src:
        txt = src.readlines()
        n_win_ds = len([True for line in txt if '\\' in line])
    return n_win_ds == 0

def vrt_win_to_linux(infile, outfile):
        # open file and replace if necessary
        with open(infile, 'r') as src:
            txt = src.readlines()
            txt_updated = [line.replace('\\', '/') for line in txt]

        with open(outfile, 'w') as tgt:
            tgt.writelines(txt_updated)

def vrt_transform_win_to_linux(vrt_file, backup=False):
    
    # check if already linux vrt file
    if not vrt_is_linux(vrt_file):
        
        # create name
        vrt_file_updated = vrt_file.parent / (vrt_file.stem + '_new.vrt')
        vrt_file_backup = vrt_file.parent / (vrt_file.stem + '_backup.vrt')
        
        # open file and replace if necessary
        vrt_win_to_linux(vrt_file, vrt_file_updated)

        # renaming
        if backup:
            shutil.copy2(vrt_file, vrt_file_backup)
        os.remove(vrt_file)
        os.rename(vrt_file_updated, vrt_file)
        


# In[3]:


import rasterio

def vrt_children_exist(vrt_file):
    with rasterio.open(vrt_file, 'r') as src:
        return all([Path(f).exists() for f in src.files]) and len(src.files) > 1


# In[4]:


def get_vrt_children_exist(row):
    out = [vrt_children_exist(row['products_dir'] / item)  for item in ['Ortho.vrt', 'DSM.vrt']]
    return pd.Series(out, index=['vrt_chilren_Ortho_exists', 'vrt_chilren_DSM_exists'], name=row.name)


# In[5]:


def calculate_aoi_area(row):
    filename = DIR_AOI / (row.project_name + '.geojson') 
    gdf = gpd.read_file(filename).to_crs(epsg=32604)
    return (gdf.geometry.area / 1e6).round(2).sum()


# ## Setup 
# * paths

# In[6]:


# setup basepaths
DIR_BASE = Path(r'S:\p_macsprocessing')
DIR_DATA_PRODUCTS = DIR_BASE / 'data_products'
DIR_AOI = DIR_BASE / 'aoi'


# In[7]:


# check if directories all exists
for d in [DIR_BASE, DIR_DATA_PRODUCTS, DIR_AOI]:
    assert d.exists()


# ## Calculate Statistics 
# * Files
# * File Count
# * file count accross types
# * aoi (size?)
# 

# #### Setup basic Dataframe and split input name

# In[8]:


df = pd.DataFrame(columns=['project_name', 'products_dir'])
# create pathlist of output products
dir_list = list(DIR_DATA_PRODUCTS.glob('*'))
df['products_dir'] = dir_list
# get project name
df['project_name'] = df['products_dir'].apply(lambda x: x.name)
# add site specific details
df = split_name_details(df)


# In[9]:


#check ortho, dsm and processing_info
file_check_columns = ['DSM', 'Ortho','processing_info']
cols_file_check = flatten([[f"{item}_dir_exists", f"{item}_n_files"] for item in file_check_columns])

file_check_output = df.apply(file_check, dirs=file_check_columns, axis=1)
df = df.join(pd.DataFrame(file_check_output.to_list(), columns=cols_file_check))


# In[10]:


# check if aoi exists
df['aoi_exists'] = df.apply(lambda x: (DIR_AOI / f'{x.project_name}.geojson').exists(), axis=1)
df['aoi_area_km2'] = df.apply(calculate_aoi_area, axis=1)


# In[11]:


# check point cloud files
PC_files = df.iloc[:].apply(file_check_PC, dirs=['PointClouds'], axis=1)
df = df.join(pd.DataFrame(PC_files.to_list(), columns=['PointCloudsRGB_n_files', 'PointCloudsNIR_n_files']))
# get mean point cloud density
df = df.join(df.iloc[:].apply(get_median_point_density, axis=1))


# In[12]:


#check for base files
# has vrt files
df['vrt_exists'] = df.apply(check_files_vrt, axis=1)
# has previews
df['previews_exists'] = df.apply(check_files_previews, axis=1)
# has previews
df['footprints_exists'] = df.apply(check_files_footprints, axis=1)


# In[13]:


df['vrt_is_linux'] = df.apply(check_vrt_is_linux, axis=1)


# #### Vrt can read all files

# In[14]:


out_tmp = df.apply(get_vrt_children_exist, axis=1)
df = df.join(out_tmp)


# #### File counts in subdirs 

# In[15]:


df = check_file_count(df)


# In[16]:


df.head()


# ## Export
# * colored df
# * csv
# * pdf
# * excel?

# #### Create styling by column

# In[17]:


df['valid_count_dsm_ortho_equal'] = df['DSM_n_files'] == df['Ortho_n_files']
df['valid_count_pcrgb_pcnir_equal'] = df['PointCloudsRGB_n_files'] == df['PointCloudsNIR_n_files']
#df['valid_count_pc_raster_equal']


# In[18]:


# create subsets for styling
subset_round = [s for s in df.columns if s.endswith('_density')] + ['aoi_area_km2']
subset_cols = [s for s in df.columns if s.endswith('n_files')] + subset_round
subset_exists = [s for s in df.columns if s.endswith('_exists')]
subset_valid_counts =  [s for s in df.columns if s.startswith('valid_count_')]
subset_valid_styler = ['project_name', 'products_dir', 'all_valid']


# In[19]:


df['all_valid'] = df[subset_exists + subset_valid_counts].all(axis=1) 


# In[20]:


df_styled = df.style.background_gradient(cmap='Blues', subset=subset_cols[:], axis=0).background_gradient(cmap='Greens', subset=subset_exists, axis=0, vmin=0, vmax=1).applymap(highlight_zero).apply(highlight_invalid, axis=1, subset=subset_valid_styler).format('{:.2f}', subset=subset_round)


# In[21]:


df.to_pickle(DIR_BASE / 'processing_status_report_raw.z')
df_styled.to_html(DIR_BASE / 'processing_status_report.html')


# In[ ]:





# ## Aggregated statistics
# 

# In[22]:


df_group_sums = df.groupby('site').sum()[['Ortho_n_files', 'aoi_area_km2', 'aoi_exists']]
df_group_sums['n_tiles'] = (df_group_sums['Ortho_n_files'] / 2).astype(int)


# In[23]:


df_group_means = df.groupby('site').mean()[['PointCloudsRGB_density', 'PointCloudsNIR_density']]


# In[24]:


df_group_first = df.groupby('site').first()[['date', 'spatial_resolution', 'region']]
df_group_first['date'] = pd.to_datetime(df_group_first['date'])


# In[25]:


df_grouped = pd.concat([df_group_first, df_group_sums, df_group_means, ], axis=1).sort_values(by='date').reset_index(drop=False).drop(columns=['Ortho_n_files']).round(2)
df_grouped_final = df_grouped.rename(columns={'aoi_exists':'Number of subprojects', 'site':'Target Name', 'date':'Date','aoi_area_km2':'Area covered km²'})


# In[26]:


df_grouped_final.to_pickle(DIR_BASE / 'processing_status_report_grouped_raw.z')
df_grouped_final.to_html(DIR_BASE / 'processing_status_report_grouped.html')


# In[ ]:




