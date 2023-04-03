import itertools
import pandas as pd
from pathlib import Path


def flatten(lst):
    return list(itertools.chain.from_iterable(lst))


def split_name_details(df):
    """
    Splits the project_name column of the input dataframe into multiple columns.

    Args:
        df (pandas.DataFrame): Input dataframe with a project_name column.

    Returns:
        pandas.DataFrame: A new dataframe with columns 'region', 'site', 'date', 'spatial_resolution', and 'subset'.
    """
    return df.join(pd.DataFrame(df.project_name.str.split('_').to_list(), columns=['region', 'site', 'date', 'spatial_resolution', 'subset']))


def file_check(row, dirs=['DSM', 'Ortho', 'processing_info'], extensions=['*.tif', '*.tif.ovr','*.log', '*_nav.txt', '*_report.pdf']):
    """
    Check if the specified directories and files exist.

    Args:
        row (pandas.Series): A row of a pandas DataFrame containing the following columns:
            - products_dir (pathlib.Path): The path to the directory containing the data.

    Returns:
        list: A list of boolean values indicating whether each directory and file exists. The list contains
        the following elements in order:
            - True if the DSM directory exists, False otherwise.
            - The number of files with a .tif, .tif.ovr, .las, .log, _nav.txt or _report.pdf extension in the DSM directory.
            - True if the Ortho directory exists, False otherwise.
            - The number of files with a .tif, .tif.ovr, .las, .log, _nav.txt or _report.pdf extension in the Ortho directory.
            - True if the PointClouds directory exists, False otherwise.
            - The number of files with a .tif, .tif.ovr, .las, .log, _nav.txt or _report.pdf extension in the PointClouds directory.
            - True if the processing_info directory exists, False otherwise.
            - The number of files with a .tif, .tif.ovr, .las, .log, _nav.txt or _report.pdf extension in the processing_info directory.
    """
    outcols = []
    for d in dirs:
        data_dir = (row['products_dir'] / d)
        has_dir = data_dir.exists()
        outcols.append(has_dir)
        ex = [list(data_dir.glob(e)) for e in extensions]        
        n_files = len(flatten(ex))
        outcols.append(n_files)
    return outcols


def file_check_PC(row, dirs=['PointClouds'], extensions=[ 'PointCloudRGB', 'PointCloudNIR']):
    outcols = []
    for d in dirs:
        data_dir = (row['products_dir'] / d)
        has_dir = data_dir.exists()
        ex = [len(list(data_dir.glob(f'*_{e}_*'))) for e in extensions]
        outcols.append(ex)
    return outcols[0]


def check_files_previews(row):
    return all([(row['products_dir'] / (row['project_name'] + f'_preview_{item}.png')).exists() for item in ['CIR', 'RGB', 'DSM']])


def check_files_vrt(row):
    return all([(row['products_dir'] / item).exists() for item in ['Ortho.vrt', 'DSM.vrt']])


def check_files_footprints(row):
    return (row['products_dir'] / (row['project_name'] + f'_tile_footprints.geojson')).exists()


def check_file_count(df):
    df['valid_count_dsm_ortho_equal'] = df['DSM_n_files'] == df['Ortho_n_files']
    df['valid_count_pcrgb_pcnir_equal'] = df['PointCloudsRGB_n_files'] == df['PointCloudsNIR_n_files']
    df['valid_count_pc_raster_equal'] = df['PointCloudsRGB_n_files']*2 == df['Ortho_n_files']
    return df


def color_negative_red(val):
    if not isinstance(val, float):
        return ''
    else:
        color = 'red' if val > 0 else 'white'
        return f'background-color: {color}'
    
    
def color_orange(val):
    if not isinstance(val, float):
        return ''
    else:
        color = 'orange' if val == 0 else 'white'
        return f'background-color: {color}'
    
    
def highlight_zero(val):
    color = 'red' if val == 0 else None
    return f'background-color: {color}'


def highlight_invalid(row):
    color = '#FFA500' if row['all_valid'] == False else 'white'
    return [f'background-color: {color}' for _ in row]