import argparse
import shutil

from joblib import Parallel, delayed

from utils_postprocessing import *
from tqdm import tqdm

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_file", type=Path,
                    required=False,
                    help="input archive file")

parser.add_argument("-p", "--processing_dir", type=Path, default=Path(r'E:\MACS_Batch_Processing'),
                    help="Parent directory where to unpack files")

parser.add_argument("-n", "--dataset_name", type=str, default=None,
                    help="Preselect dataset id.")

parser.add_argument("-o", "--output_dir", type=Path, default=Path(r'S:\p_macsprocessing\data_products'),
                    help="Main product directory")

parser.add_argument("-d", "--directory",
                    default=[Path(r'D:\MACS_Batch_Processing'),
                             Path(r'E:\MACS_Batch_Processing'),
                             Path(r'G:\MACS_Data_Storage'),
                             Path(r'F:\MACS_Data_Storage')
                             ],
                    type=list,
                    help="data directories")

args = parser.parse_args()
dir_list = args.directory
if not isinstance(dir_list, list):
    dir_list = list(dir_list)


def main():
    """
    # 3. move files to product dir

    # 4. delete intermediate files
    """
    if not args.input_file:
        for process_dir in dir_list:
            flist = list(process_dir.glob(f'*{args.dataset_name}*'))
            if len(flist) > 0:
                p_file = flist[0]
                continue
            else:
                print("No matches found")
                continue
    else:
        p_file = args.input_file

    site_name = p_file.stem
    unzip_tarfile(p_file, site_name, target_dir=args.processing_dir)

    # setup paths
    ortho_product_list, point_cloud_dir, point_cloud_nir, point_cloud_rgb = setup_PCclip_paths(site_name)

    # run clipping RGB
    print('Start clipping point clouds')
    Parallel(n_jobs=40)(
        delayed(clip_las_file)(ortho_file, point_cloud_rgb, point_cloud_dir, product_name='PointCloudRGB') for ortho_file in tqdm(ortho_product_list[:]))
    Parallel(n_jobs=40)(
        delayed(clip_las_file)(ortho_file, point_cloud_nir, point_cloud_dir, product_name='PointCloudNIR') for
        ortho_file in
        tqdm(ortho_product_list[:]))

    final_dir = args.output_dir / site_name / '06_DataProducts'
    shutil.move(point_cloud_dir, final_dir)

    return 0


def setup_PCclip_paths(site_name):
    data_products_dir = args.processing_dir / site_name / '06_DataProducts'
    footprints_file = list(data_products_dir.glob('*.geojson'))[0]
    #
    raster_products_dir = args.output_dir / site_name
    ortho_product_list = list((raster_products_dir / 'Ortho').glob(f'{site_name}*.tif'))
    point_cloud_dir = args.processing_dir / site_name / '04_pix4d' / site_name / '2_densification' / 'point_cloud'
    point_cloud_rgb = list(point_cloud_dir.glob('*group1_densified_point_cloud*.las'))[0]
    point_cloud_nir = list(point_cloud_dir.glob('*NIR_densified_point_cloud*.las'))[0]
    # create tiles
    point_cloud_dir = args.processing_dir / site_name / '06_DataProducts' / 'PointClouds'
    os.makedirs(point_cloud_dir, exist_ok=True)
    return ortho_product_list, point_cloud_dir, point_cloud_nir, point_cloud_rgb


def clip_las_file(ortho_file, point_cloud, point_cloud_dir, product_name='PointCloudRGB'):
    outfile = ortho_file.stem.replace('Ortho', product_name) + '.las'
    point_cloud_tile = point_cloud_dir / outfile
    with rasterio.open(ortho_file) as src:
        min_x, min_y, max_x, max_y = src.bounds
        s = f'las2las -keep_xy {min_x} {min_y} {max_x} {max_y} -i {point_cloud} -o {point_cloud_tile}'
        os.system(s)


if __name__ == "__main__":
    main()
