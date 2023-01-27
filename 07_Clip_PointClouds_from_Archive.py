import argparse
from utils_postprocessing import *

# warnings.filterwarnings('ignore')
from utils_postprocessing import unzip_tarfile

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
    # 2. read footprints file, get bounds, las2las

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

    return 0


if __name__ == "__main__":
    main()

