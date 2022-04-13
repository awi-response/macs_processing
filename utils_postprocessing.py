import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pathlib import Path
import rasterio
import shutil


def flist_to_df(filelist):
    """
    Create pandas DataFrame with information parsed from filelist

    Parameters
    ----------
    filelist : list
        file list of MACS Orthomosaic Output files

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    sensors = []
    rows = []
    cols = []
    for f in filelist:
        sensor, row, col = f.name[:-4].split('_')[-3:]
        sensors.append(sensor)
        rows.append(row)
        cols.append(col)

    df = pd.DataFrame(columns=['filename', 'sensor', 'row', 'col'])
    df['filename'] = filelist
    df['sensor'] = sensors
    df['row'] = rows 
    df['col'] = cols 
    return df

# create more specific vrt files for each run to avoid duplicates on parallel run
def stack_output(outmosaic, rgbfile, nirfile, remove_temporary_files=True):
    """
    Function to stack together RGB and NIR images
    Input: 
        4 Band RGB (RGB-A)
        2 Band NIR (NIR-A)
    """

    basename_rgb = rgbfile.name[:-4]
    basename_nir = nirfile.name[:-4]
    
    b1 = f'{basename_rgb}_1.vrt'
    b2 = f'{basename_rgb}_2.vrt'
    b3 = f'{basename_rgb}_3.vrt'
    b_nir = f'{basename_nir}_1.vrt'
    mos = f'{basename_rgb}.vrt'
    
    for band in [1,2,3]:
        s = f'gdalbuildvrt -b {band} {basename_rgb}_{band}.vrt {rgbfile}'
        os.system(s)

    for band in [1]:
        s = f'gdalbuildvrt -b {band} {basename_nir}_{band}.vrt {nirfile}'
        os.system(s)

    s = f'gdalbuildvrt -separate {mos} {b3} {b2} {b1} {b_nir}'
    os.system(s)

    s = f'gdal_translate -a_nodata 0 -co COMPRESS=DEFLATE -co BIGTIFF=YES {mos} {outmosaic}'
    os.system(s)
    if remove_temporary_files:
        for file in [b1, b2, b3, b_nir, mos]:
            os.remove(file)
            

def calculate_pyramids(rasterfile):
    """
    Function to calculate pyramids

    Parameters
    ----------
    rasterfile : Path
        file for which to create pyramids

    Returns
    -------
    None.

    """
    addo = f'gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE --config GDAL_NUM_THREADS ALL_CPUS {rasterfile}'
    os.system(addo)


def mask_and_name_bands(mosaic_file):
    """
    Function to mask incomplete spectral data (e.g. with only NIR data and no RGB and vice versa)
    Add names to Bands
    """
    with rasterio.open(mosaic_file, 'r+') as src:
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


def full_postprocessing_optical(df, tile_id, rgb_name='group1', nir_name='nir'):
    """
    Wrapper function to sequentially run

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    tile_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    subset = df[df['tile_id'] == tile_id]
    rgbfile = subset.query(f'sensor=="{rgb_name}"').filename.values[0]
    nirfile = subset.query(f'sensor=="{nir_name}"').filename.values[0]
    outmosaic = rgbfile.parent / f'mosaic_{tile_id}.tif'
    stack_output(outmosaic, rgbfile, nirfile, remove_temporary_files=True)
    calculate_pyramids(outmosaic)
    mask_and_name_bands(outmosaic)
    

def move_and_rename_processed_tiles(df, out_basename, target_dir, product_type, move=False):
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe created by flist_to_df().
    out_basename : str
        basename of outputfile.
    target_dir : str/Path
        output directory.
    product_type : str
        simple name identifier e.g. 'Ortho' or 'DSM'.
    move : bool, optional
        set True if files should be moved instead of copied. The default is False.

    Returns
    -------
    None.

    """
    for index, row in df.iloc[:].iterrows():
        tile_id = row.tile_id
        infile = row.filename
        infile_ovr = Path(str(infile) + '.ovr')
        outfile = target_dir / f'{out_basename}_{product_type}_{tile_id}.tif'
        outfile_ovr = Path(str(outfile) + '.ovr')
        #print(infile, outfile)
        
        if move:
            shutil.move(infile, outfile)
        else:
            shutil.copy(infile, outfile)
        try:
            if move:
                shutil.move(infile_ovr, outfile_ovr)
            else:
                shutil.copy(infile_ovr, outfile_ovr)
        
        except Exception as e:
            print(e)


def create_mask_vector(raster_file, temporary_target_dir, remove_raster_mask=False, polygonize = Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'):
    """
    Function to create vectors of valid data for a raster file
    Parameters
    ----------
    raster_file : Path
        input raster file from which to create vector mask.
    temporary_target_dir : Path
        directory path where temporary files should be stored.
    remove_raster_mask : bool, optional
        delete temporary raster mask. The default is False.
    polygonize : Path, optional
        Path to gdal polygonize function, Automatically created for (Windows) conda environments. The default is Path(os.environ['CONDA_PREFIX']) / 'Scripts' / 'gdal_polygonize.py'.

    Returns
    -------
    mask_vector : Path
        path of output vector footprints file.

    """
    maskfile = temporary_target_dir / (raster_file.stem + '_mask.tif')
    mask_vector = temporary_target_dir / (raster_file.stem + '_mask.geojson')

    s_extract_mask = f'gdal_translate -ot Byte -b mask {raster_file} {maskfile}'
    os.system(s_extract_mask)

    s_polygonize_mask = f'{polygonize} -f GeoJSON {maskfile} {mask_vector}'
    os.system(s_polygonize_mask)
    
    # needs to be fixed - is not deleting at the moment
    if remove_raster_mask:
        os.remove(maskfile)
    
    return mask_vector

def load_and_prepare_footprints(vector_file):
    """
    

    Parameters
    ----------
    vector_file : Path
        OGR compatible vector file (basic footprint).

    Returns
    -------
    gdf : TYPE
        filled GeoDataframe.

    """
    gdf = gpd.read_file(vector_file)
    file_name_raster = Path(str(vector_file).replace('_mask.geojson', '.tif')).name
    gdf['Orthomosaic'] = file_name_raster
    gdf['DSM'] = file_name_raster.replace('_Ortho_', '_DSM_')
    return gdf

def merge_single_vector_files(gdf_list, outfile, site_name, date_local):
    """
    Function to merge a list of Geodataframes (individual image footprints) \
    to a larger one.
    
    Parameters
    ----------
    gdf_list : list
        List of GeoDataframes of identical type.
    outfile : Path
        path of merged GeoDataframe.
    site_name : str
        basename of site identifier.
    date_local : str
        date in 'YYYY-MM-DD' format.

    Returns
    -------
    None.

    """
    # Merge and save final footprints file
    gdf_merged = gpd.GeoDataFrame(pd.concat(gdf_list))
    gdf_merged = gdf_merged.set_crs(crs=gdf_list[0].crs).to_crs(epsg=4326)

    gdf_merged = gdf_merged[gdf_merged['DN'] > 0]
    gdf_merged = gdf_merged.drop(columns=['DN'])

    gdf_merged['Site_name'] = site_name
    gdf_merged['Date'] = date_local#f'{date_local[:4]}-{date_local[4:6]}-{date_local[6:]}'

    gdf_merged.to_file(outfile)
    
    
def parse_site_name(site_name):
    region, site, site_number, date_tmp, resolution = site_name.split('_')
    date = f'{date_tmp[:4]}-{date_tmp[4:6]}-{date_tmp[6:]}'
    return region, site, site_number, date, resolution
