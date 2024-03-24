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

args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###


def main():
    product_dir = Path(settings.PROJECT_DIR)
    site_name = product_dir.name
    assert product_dir.exists()

    tar_file_name = product_dir.parent / f'{product_dir.name}.tar'
    assert not tar_file_name.exists()

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
