
# coding: utf-8

# ### Script to convert geolocation into correct format

# In[1]:


#import geopandas as gpd
import pandas as pd
import os


INFILE = 'nav.txt'
outfile = 'geo_pix4d_new.txt'
#NEW_CRS = 'EPSG:4326'
H_ACC = 1
V_ACC = 1


# #### Load images 
df = pd.read_csv(INFILE, sep='\t')

# #### Change image suffixes 
images = df['File '].str.replace('.macs', '.tif')
images = images.apply(lambda x: x.strip().split('/')[-1])

# #### Fill Table 
df['imagename_tif'] = images
df['x'] = df['Lon[deg] ']
df['y'] = df['Lat[deg] ']
df['horizontal_accuracy'] = H_ACC
df['vertical_accuracy'] = V_ACC

# #### Create final structure
df_new = df[['imagename_tif', 'y', 'x', 'Alt[m] ', 'Omega[deg] ',
       'Phi[deg] ', 'Kappa[deg]', 'horizontal_accuracy', 'vertical_accuracy']]

# #### Export file 
df_new.to_csv(outfile, sep='\t', header=True, index=False)