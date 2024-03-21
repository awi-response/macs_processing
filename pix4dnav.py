# coding: utf-8
# ### Script to convert geolocation into correct format

import argparse
import pandas as pd
import os


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--infile", type=str, default='nav.txt',
                    help="input file")
parser.add_argument("-o", "--outfile", type=str, default='geo_pix4d_new.txt',
                    help="output file")
parser.add_argument("-ha", "--horizontal_accuracy", type=float, default=1.0,
                    help="Horizontal accuracy for pix4D. Default = 1")
parser.add_argument("-va", "--vertical_accuracy", type=float, default=1.0,
                    help="Horizontal accuracy for pix4D. Default = 1")

args = parser.parse_args()
INFILE = args.infile#'nav.txt'
outfile = args.outfile#'geo_pix4d_new.txt'
H_ACC = args.horizontal_accuracy
V_ACC = args.vertical_accuracy


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