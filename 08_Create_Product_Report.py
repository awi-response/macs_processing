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

# In[ ]:


import pandas as pd
from pathlib import Path
import itertools
import shutil

from utils_report import *


# In[ ]:


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


# ## Setup 
# * paths

# In[ ]:


# setup basepaths
DIR_BASE = Path(r'S:\p_macsprocessing')
DIR_DATA_PRODUCTS = DIR_BASE / 'data_products'
DIR_AOI = DIR_BASE / 'aoi'


# In[ ]:


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

# In[ ]:


df = pd.DataFrame(columns=['project_name', 'products_dir'])
# create pathlist of output products
dir_list = list(DIR_DATA_PRODUCTS.glob('*'))
df['products_dir'] = dir_list
# get project name
df['project_name'] = df['products_dir'].apply(lambda x: x.name)
# add site specific details
df = split_name_details(df)


# In[ ]:


#check ortho, dsm and processing_info
file_check_columns = ['DSM', 'Ortho','processing_info']
cols_file_check = flatten([[f"{item}_dir_exists", f"{item}_n_files"] for item in file_check_columns])

file_check_output = df.apply(file_check, dirs=file_check_columns, axis=1)
df = df.join(pd.DataFrame(file_check_output.to_list(), columns=cols_file_check))


# In[ ]:


# check if aoi exists
df['aoi_exists'] = df.apply(lambda x: (DIR_AOI / f'{x.project_name}.geojson').exists(), axis=1)


# In[ ]:


# check point cloud files
PC_files = df.iloc[:].apply(file_check_PC, dirs=['PointClouds'], axis=1)
df = df.join(pd.DataFrame(PC_files.to_list(), columns=['PointCloudsRGB_n_files', 'PointCloudsNIR_n_files']))


# In[ ]:


#check for base files
# has vrt files
df['vrt_exists'] = df.apply(check_files_vrt, axis=1)
# has previews
df['previews_exists'] = df.apply(check_files_previews, axis=1)
# has previews
df['footprints_exists'] = df.apply(check_files_footprints, axis=1)


# #### TODO: check vrt for linux style paths
# * open vrt
# * scan for '\' in paths
# * df['vrt_Ortho_is_linux'] = ...
# * df['vrt_DSM_is_linux'] = ...

# In[ ]:


df['vrt_is_linux'] = df.apply(check_vrt_is_linux, axis=1)


# #### File counts in subdirs 

# In[ ]:


df = check_file_count(df)


# In[ ]:


df.head()


# ## Export
# * colored df
# * csv
# * pdf
# * excel?

# #### Create styling by column

# In[ ]:


df['valid_count_dsm_ortho_equal'] = df['DSM_n_files'] == df['Ortho_n_files']
df['valid_count_pcrgb_pcnir_equal'] = df['PointCloudsRGB_n_files'] == df['PointCloudsNIR_n_files']
df['valid_count_pc_raster_equal']


# In[ ]:


subset_cols = [s for s in df.columns if s.endswith('n_files')]
subset_exists = [s for s in df.columns if s.endswith('_exists')]
subset_valid_counts =  [s for s in df.columns if s.startswith('valid_count_')]
subset_valid_styler = ['project_name', 'products_dir', 'all_valid']


# In[ ]:


df['all_valid'] = df[subset_exists + subset_valid_counts].all(axis=1) 


# In[ ]:


df_styled = df.style.background_gradient(cmap='Blues', subset=subset_cols[:], axis=0).background_gradient(cmap='Greens', subset=subset_exists, axis=0, vmin=0, vmax=1).applymap(highlight_zero).apply(highlight_invalid, axis=1, subset=subset_valid_styler)


# In[ ]:


df_styled


# In[ ]:


df_styled.to_html(DIR_BASE / 'processing_status_report.html')

