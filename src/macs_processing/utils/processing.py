# -*- coding: utf-8 -*-
"""
MACS_Processing utils
"""
import os, shutil, rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from skimage import morphology
from joblib import delayed, Parallel
from scipy.stats import linregress
import tqdm


# Functions for MACS to TIF

def prepare_df_for_mipps2(path_footprints, path_infiles):
    # Load filtered footprints files
    df = gpd.read_file(path_footprints)

    flist = list(Path(path_infiles).glob('*/*.macs'))
    flist = [f'"{str(f)}"' for f in flist]

    df_full = pd.DataFrame()
    df_full['full_path'] = flist
    df_full['basename'] = pd.DataFrame(df_full['full_path'].apply(lambda x: os.path.basename(x)))
    # return Inner join of lists - create filtered list of filepaths 
    return df.set_index('Basename').join(df_full.set_index('basename'))


def prepare_df_for_mipps(path_footprints, path_infiles):
    # Load filtered footprints files
    df = gpd.read_file(path_footprints)

    # flist = glob.glob(path_infiles + '/*/*.macs')
    flist = list(Path(path_infiles).glob('*/*.macs'))
    flist = [str(f) for f in flist]

    df_full = pd.DataFrame()
    df_full['full_path'] = flist
    df_full['basename'] = pd.DataFrame(df_full['full_path'].apply(lambda x: os.path.basename(x)))
    # workaroud for different versions
    if 'Basename' in df.columns:
        df.set_index('Basename', inplace=True)
    elif 'Name' in df.columns:
        df.set_index('Name', inplace=True)

    df_full['basename'] = pd.DataFrame(df_full['full_path'].apply(lambda x: os.path.basename(x)))
    # return Inner join of lists - create filtered list of filepaths
    return df.join(df_full.set_index('basename'))


def write_exif(outdir, tag, exifpath):
    s = f'{exifpath} -overwrite_original -Model="{tag}" {outdir}'
    print(s)
    os.system(s)


# Functions for image cropping and masking

def make_mask(shape, disksize=4864):
    dsk = morphology.disk((disksize - 1) / 2)

    diff = np.array(dsk.shape) - np.array(shape)
    r_start = round(diff[0] / 2)
    r_end = r_start + shape[0]
    c_start = round(diff[1] / 2)
    c_end = c_start + shape[1]
    cropped_dsk = dsk[r_start:r_end, c_start:c_end]
    return cropped_dsk


def mask_and_tag(image, mask, tag=None):
    # mask3 = np.r_[[mask]*n_bands]
    with rasterio.open(image, mode='r+') as src:
        if tag:
            src.update_tags(Model=tag)
            print(src.tags)
        src.profile.update(
            nodata=0,
            compress='lzw')
        data = src.read() * mask
        src.write(data)
    newimage = f'{str(image)[:-4]}_new.tif'
    os.system(f'gdal_translate -a_nodata 0 {str(image)} {str(newimage)}')
    os.remove(str(image))
    shutil.move(str(newimage), str(image))


# SCALING functions
def rescale(array, minimum, maximum, gain=1.):
    x = [0, 2 ** 16 - 1]
    y = [minimum, maximum]
    slope, intercept, r_value, p_value, std_err = linregress(y, x)
    # print(slope,intercept)
    D = (array * gain) * slope + intercept
    D_round = np.around(np.clip(D, 1, 2 ** 16 - 1))
    return np.array(D_round, np.uint16)


def write_new_values(image, minimum, maximum, shutter_factor=1, tag=True):
    with rasterio.open(image, mode='r+') as src:
        data = src.read()
        datanew = rescale(data, minimum, maximum, gain=shutter_factor)
        src.write(datanew)
        if tag:
            src.update_tags(VALUE_STRETCH_MINIMUM=minimum, VALUE_STRETCH_MAXIMUM=maximum)


def read_stats(image):
    with rasterio.open(image) as src:
        a = src.read()
        return a.mean(), a.min(), a.max(), a.std()


def read_stats_extended(image):
    with rasterio.open(image) as src:
        a = src.read()
        return a.mean(), a.min(), a.max(), a.std(), np.percentile(a, 1), np.percentile(a, 99)


def get_image_stats_multi(OUTDIR, sensors, n_jobs=40, nth_images=1, max_images=3000, quiet=False):
    dfs = []
    for sensor in sensors:
        imlist = list(OUTDIR[sensor].glob('*.tif'))
        if len(imlist) > max_images:
            images = list(np.random.choice(imlist, size=max_images))#####
        else:
            images = imlist[::nth_images]
        # skip empt
        if len(images) == 0:
            continue
        if quiet:
            stats = Parallel(n_jobs=n_jobs)(delayed(read_stats_extended)(image) for image in images)
        else:
            stats = Parallel(n_jobs=n_jobs)(delayed(read_stats_extended)(image) for image in tqdm.tqdm_notebook(images))
        df_stats = pd.DataFrame(data=np.array(stats), columns=['mean', 'min', 'max', 'std', 'p01', 'p99'])
        df_stats['image'] = images
        df_stats['sensor'] = sensor
        dfs.append(df_stats)
    df = pd.concat(dfs)
    return df


def get_shutter_factor(OUTDIR, sensors):
    """
    get shutter ratio between RGB and NIR. scaling factor for RGB (>1 = increase in values) 
    """
    shutter = {}
    for sensor in sensors:
        images = list(OUTDIR[sensor].glob('*.tif'))
        if len(images) == 0:
            continue
        f = images[0]
        shutter[sensor] = int(f.stem.split('_')[-1])
    if ('right' in sensors) and ('nir' in sensors):
        factor = shutter['nir'] / shutter['right']
    elif ('left' in sensors) and ('nir' in sensors):
        factor = shutter['nir'] / shutter['left']
    else:
        factor = 1
    return factor


# Functions for footprints creation 

def get_overlapping_ds(aoi_path, projects_file, name_attribute='Dataset'):
    # Open Projects file and AOI
    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    if 'Dataset' in aoi.columns: aoi.drop(columns=['Dataset'], inplace=True)
    datasets = gpd.read_file(projects_file).to_crs(epsg=4326)
    overlapping_ds = gpd.sjoin(datasets, aoi, lsuffix='', how='inner').drop_duplicates(subset=name_attribute)
    return overlapping_ds


def retrieve_footprints(overlapping_ds, project_id, parent_data_dir, aoi_file, fp_file_regex='*full.shp',
                        name_attribute='Dataset'):
    try:
        project_name = overlapping_ds.loc[project_id][name_attribute]
    except:
        project_name = overlapping_ds.loc[int(project_id)][name_attribute]

    if isinstance(project_name, pd.Series):
        project_name = project_name.iloc[0]

    # crashes in 2023 configuration
    # streamline structure for all projects
    footprint_list = list((parent_data_dir / project_name).glob(fp_file_regex))
    if len(footprint_list) == 0:
        footprint_list = list((parent_data_dir / project_name).glob(fp_file_regex.replace('.shp', '.gpkg')))
    footprints = footprint_list[0]
    fp = gpd.read_file(footprints).to_crs(epsg=4326)
    aoi = gpd.read_file(aoi_file).to_crs(epsg=4326)[['geometry']]

    fp_selection = gpd.sjoin(fp, aoi)
    return fp_selection


def get_dataset_name(ds, dataset_id, name_attribute='Dataset'):
    """

    """
    try:
        dataset_name = ds.loc[int(dataset_id)][name_attribute]
    except:
        dataset_name = ds.loc[dataset_id][name_attribute]
    return dataset_name


def get_dataset_stats(datasets_file, parent_dir, aoi, name_attribute='Dataset'):
    grp = []
    idxs = datasets_file[name_attribute].index
    for idx in idxs:
        dataset_name = datasets_file[name_attribute].loc[idx]
        footprints = retrieve_footprints(datasets_file, idx, parent_dir, aoi)
        
        # workaround for missing 'Looking' attribute
        if 'Looking' in footprints.columns:
            macs_config = 'MACS2018'
            stats = footprints.groupby(by='Looking').count().iloc[:, 0].T
            for col in ['center', 'left', 'right']:
                if col not in stats.index:
                    stats[col] = 0
            stats['total_images'] = stats[['center', 'left', 'right']].sum()
        else:
            stats = footprints.groupby(by='Sensor').count().iloc[:, 0].T
            for col in ['RGB', 'NIR']:
                if col not in stats.index:
                    stats[col] = 0
            stats['total_images (NIR+RGB)'] = stats[['RGB', 'NIR']].sum()

        grp.append(stats.rename(dataset_name))
    stats = pd.concat(grp, axis=1).T
    
    stats['dataset_id'] = idxs
    stats[name_attribute] = stats.index
    stats = stats.set_index('dataset_id', drop=True)
    
    return stats
