import os
from pathlib import Path
import zipfile, tarfile
import tqdm
import pandas as pd
import argparse
import shutil
from utils_postprocessing import *

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--archive_dir", type=Path,
                    default=Path(r'\\hssrv1.awi.de\projects\p_macsprocessing\backup_processing_files'),
                    help="Directory with archived files")

parser.add_argument("-t", "--target_dir", type=Path, default=Path(r'E:\MACS_Batch_Processing'),
                    help="Parent directory where to unpack files")

parser.add_argument("-l", "--list_files", action='store_true',
                    help="List only datasets without processing")

parser.add_argument("-d", "--dataset_name", type=str, default=None,
                    help="Preselect dataset id.")

parser.add_argument("-r", "--rawdata", action='store_true',
                    help="Select if you want to extract the raw data.")

args = parser.parse_args()
print(args)

tape_path = args.archive_dir
processing_path = args.target_dir


def main():
    if args.dataset_name:
        p_file = tape_path / f'{args.dataset_name}.tar'
        if not p_file.exists():
            print(f'File: {p_file} does not exist!')
            raise ValueError
    else:
        # list tar files
        plist = list(Path(tape_path).glob('*.tar'))
        # Put into DataFrame
        df = pd.DataFrame(data=plist, columns=['tar_file'])
        df['project_name'] = df.apply(lambda x: x['tar_file'].stem, axis=1)
        df['project_file_size'] = df.apply(lambda x: (x['tar_file'].stat().st_size) / 1e9, axis=1)
        print(df[['project_name', 'project_file_size']])

        if not args.list_files:
            dataset_id = input('Please select IDs (comma separated): ')
            p_file = df.loc[int(dataset_id)]['tar_file']

    # process
    if not args.list_files:
        # print size
        stats = p_file.stat()
        print(f'Archive file size: {stats.st_size / 1e9} GB')
        disk_space = shutil.disk_usage(processing_path).free / 1e9
        print(f'Disk space left: {disk_space} GB')

        print('Opening file:', p_file)
        print('Extracting files to:', processing_path)

        #
        with tarfile.open(p_file) as f:
            flist = f.getnames()
            print(f'Number of files in archive: {len(flist)}')
            # check if rawdata should be processed
            if args.rawdata:
                subset = ['01_rawdata', '02_studysites', '04_pix4d']
            else:
                subset = ['02_studysites', '04_pix4d']

            unzip_subset = create_unziplist2(flist, subset=subset)
            unzip_subset = [f for f in unzip_subset if not (processing_path / f).exists()]
            print(f'Number of files to extract: {len(unzip_subset)}')

            print('Start extraction to:', processing_path)
            for ff in tqdm.tqdm(unzip_subset[:]):
                f.extract(ff, path=processing_path)

        print('Finished without extracting files')


if __name__ == "__main__":
    main()
