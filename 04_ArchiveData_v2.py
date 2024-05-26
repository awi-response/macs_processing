import argparse
from utils_postprocessing import *
import importlib
import tarfile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings",
                    type=Path,
                    help="Path to Settings file")
parser.add_argument("-i", "--input_dir", type=Path,
                    help="Path to Product Storage Space/Directory")
parser.add_argument("-a", "--archive_dir", type=Path,
                    default=Path(r'G:\MACS_Data_Storage'),
                    help="Directory with archived files")
parser.add_argument("-d", "--delete_indir", action='store_true',
                    help="set option to delete input dir")
parser.add_argument("-f", "--file_type", type=str, default="zip",
                    help="archive file type (zip or tar)")                    

args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###


def main():
    product_dir = Path(settings.PROJECT_DIR)
    site_name = product_dir.name
    assert product_dir.exists()

    tar_file_name = product_dir.parent / f'{product_dir.name}.{args.file_type}'
    assert not tar_file_name.exists()

    # delete raw data
    raw_data = product_dir / '01_rawdata' / 'tif'
    raw_data_dirs = ['33552_NIR', '33576_Right', '33577_Left']
    for rdd in raw_data_dirs:
        d = raw_data / rdd
        if d.exists():
            shutil.rmtree(d)
    
    # delete data_products
    dir_dp = product_dir / '06_DataProducts'
    if dir_dp.exists():
        shutil.rmtree(dir_dp)

    # delete data_products
    file_logpp = product_dir / f'{site_name}_postprocessing.log'
    if file_logpp.exists():
        os.remove(file_logpp)

    print(f'Start archiving {product_dir} to {tar_file_name}')
    with tarfile.open(tar_file_name, 'w') as tar:
        tar.add(product_dir, recursive=True, arcname=site_name)

    # delete files
    if args.delete_indir:
        shutil.rmtree(product_dir)

    # move to archive dir
    print(f'Start moving {tar_file_name} to {args.archive_dir}')
    shutil.move(str(tar_file_name), str(args.archive_dir))


if __name__ == "__main__":
    main()
