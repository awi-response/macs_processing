import os
from pathlib import Path
import rasterio
from whitebox import WhiteboxTools

wbt = WhiteboxTools()


def assign_crs_to_raster(infile, crs):
    working_dir = Path(wbt.get_working_dir())
    outfile = infile[:-4] + '_projected.tif'
    gdal_string = f'gdal_translate -a_srs {crs} -co COMPRESS=DEFLATE -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS {working_dir/infile} {working_dir/outfile} '
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


def create_point_cloud_tiles(point_cloud, footprint_tile_path, settings_file, target_dir, product_name='PCNir'):
    """
    point cloud: path to point cloud
    footprint_tile_path: path to footprint_tile (geojson)
    settings: settings object
    product_name: name of product (PCNIR for NIR PC or PCRGB for RGB Point clouds
    """

    # individual tile
    footprint = footprint_tile_path
    footprint_shp = footprint.with_suffix('.shp')
    # convert to shp
    os.system(f'ogr2ogr -f "ESRI Shapefile" {footprint_shp} {footprint}')
    # create output name
    tile_id = footprint.stem.rstrip("_mask").split('_Ortho_')[-1]
    outfile_name = f'{settings_file.PIX4d_PROJECT_NAME}_{product_name}_{tile_id}.las'
    #outfile = settings_file.TARGET_DIR_PC / outfile_name
    outfile = target_dir / outfile_name

    wbt.clip_lidar_to_polygon(
        i=point_cloud,
        polygons=footprint_shp,
        output=outfile
    )
    return 0
