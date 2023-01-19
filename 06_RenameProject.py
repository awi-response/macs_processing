#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import pandas as pd
import argparse
import sys

def create_outname(infile, string_in, string_replace):
    name = infile.name.replace(string_in, string_replace)
    return infile.parent / name


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inname", required=True, type=str, help="input name")
parser.add_argument("-o", "--outname", required=True, type=str, help="output name")
parser.add_argument("-d", "--directory",
                    default=[Path(r'D:\MACS_Batch_Processing'),
                             Path(r'E:\MACS_Batch_Processing'),
                             Path(r'S:\p_macsprocessing\aoi'),
                             Path(r'D:\Pix4D_Processing\macs_processing'),
                             Path(r'S:\p_macsprocessing\data_products')
                             ],
                    type=list, help="data directories")
parser.add_argument("-noask", "--noask", action='store_true', help="execute script without asking")
args = parser.parse_args()

dir_list = args.directory
if not isinstance(dir_list, list):
    dir_list = list(dir_list)

#process_dir = args.directory
oldname = args.inname
newname = args.outname


def main():
    for process_dir in dir_list:
        flist = list(process_dir.glob(f'**/*{oldname}*'))

        if len(flist) > 0:
            df = pd.DataFrame(data=flist, columns=['filename'])

            df['depth'] = df.apply(lambda x: len(x.filename.parts), axis=1)
            df['is_dir'] = df.apply(lambda x: x.filename.is_dir(), axis=1)

            df['filename_out'] = df['filename'].apply(create_outname, args=(oldname, newname))

            df_sorted = df.sort_values(by='depth', ascending=False)
        else:
            print("No matches found")
            continue
        print(f'Renaming files in {process_dir}')
        print(f'Number of files to be renamed: {len(df_sorted)}')

        prompt = 'y'
        if not args.noask:
            prompt = input('Execute renaming: [y/n]? ').lower()

        if prompt == 'y':
            print(f'Start renaming from pattern:\n{oldname}\n to \n{newname}')
            df_sorted.apply(lambda x: x.filename.rename(x.filename_out), axis=1)
        else:
            print('Skip renaming')

if __name__ == '__main__':
    main()
