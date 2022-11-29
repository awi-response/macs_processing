import os
from pathlib import Path
import rasterio
from whitebox import WhiteboxTools

wbt = WhiteboxTools()


def assign_crs_to_raster(infile, crs):
    working_dir = Path(wbt.get_working_dir())
    outfile = infile[:-4] + '_projected.tif'
    gdal_string = f'gdal_translate -a_srs {crs} -co COMPRESS=DEFLATE {working_dir/infile} {working_dir/outfile}'
    os.system(gdal_string)
    return outfile


def merge_point_clouds(which_point_cloud='nir'):
    working_dir = Path(wbt.get_working_dir())
    regex_filter = {'nir': '*Grayscale', 'rgb': '*group1', 'both': ''}
    regex_filter2 = {'nir': '*NIR', 'rgb': '*group1', 'both': ''}

    regex = f'{regex_filter[which_point_cloud]}*.las'
    merge_list = list(working_dir.glob(regex))

    if len(merge_list) == 0:
        regex = f'{regex_filter2[which_point_cloud]}*.las'
        merge_list = list(working_dir.glob(regex))

    output = f'merged_{which_point_cloud}.las'
    inputs = ','.join([f.name for f in merge_list if f.name not in ['merged.las', 'merged_nir.las', 'merged_rgb.las']])

    wbt.lidar_join(
        inputs=inputs,
        output=output
    )
    return output


def pc_IDW_toDSM(infile, resolution, ret='all'):
    outfile = f'{infile[:-4]}_IDW.tif'
    wbt.lidar_idw_interpolation(
        i=infile,
        output=outfile,
        parameter="elevation",
        returns=ret,
        resolution=resolution,
        weight=1.0,
        radius=2.5,
    )
    return outfile


def fill_holes(infile, filter=11):
    #filter = int(5/resolution)
    outfile = infile[:-4]+f'_filled.tif'
    wbt.fill_missing_data(
        i=infile,
        output=outfile,
        filter=filter,
        weight=2.0,
        no_edges=True
    )
    return outfile


def smooth_DSM(infile, filter=11, iterations=10, normdiff=50, max_diff=0.5):
    outfile = infile[:-4] + f'_smoothed.tif'
    wbt.feature_preserving_smoothing(
        dem=infile,
        output=outfile,
        filter=filter,
        norm_diff=normdiff,
        num_iter=iterations,
        max_diff=0.5,
        zfactor=None
    )
    return outfile


def hillshade(infile):
    outfile = infile[:-4] + '_hillshade.tif'
    working_dir = Path(wbt.get_working_dir())
    gdal_dem_string = f'gdaldem hillshade -multidirectional -co COMPRESS=DEFLATE {working_dir / infile} {working_dir / outfile}'
    os.system(gdal_dem_string)
    return outfile


def resolution_from_file(infile):
    with rasterio.open(infile) as src:
        return src.res[0]


def crs_from_file(infile):
    with rasterio.open(infile) as src:
        return src.crs.to_string()


def clip_to_tile(input_mosaic, example_tile, target_dir, rename=['transparent_mosaic_group1', 'dsm']):
    with rasterio.open(example_tile) as src:
        # get image properties
        bounds = src.bounds
        resolution = src.res
        # file names
        rename=['transparent_mosaic_group1', 'dsm']
        stem = example_tile.stem
        stem_out = stem.replace(rename[0], rename[1])
        clipped = target_dir / f'{stem_out}.tif'
        # run gdal_translate
        # run with pixel count
        gdal_string = f'gdalwarp -te {bounds.left} {bounds.top} {bounds.right} {bounds.bottom} -ts {src.width} {src.height} -co COMPRESS=DEFLATE {input_mosaic} {clipped}'
        os.system(gdal_string)
