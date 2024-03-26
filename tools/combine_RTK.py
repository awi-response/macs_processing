from pathlib import Path
import argparse

# add argparsing option

parser = argparse.ArgumentParser(description="Merge RTK files from individual sensors")
parser.add_argument("--indir", default=".", help="Input directory (default: current directory)")
parser.add_argument("--regex", default="*RTK*.txt", help="regular expression to find input files (default: '*RTK*.txt')")
args = parser.parse_args()
indir = Path(args.indir)
regex = args.regex

# find input files
flist = list(indir.glob(regex))

with open(flist[0]) as src:
    header = [src.readline()]
for f in flist:
    with open(f) as src:
        data = src.readlines()[1:]
    header += data

basename = flist[0].absolute().parent.name
outfile = flist[0].absolute().parent / f'{basename}_nav_RTK.txt'
print(f'Merging navfiles to {outfile}')
with open(outfile, 'w') as dst:
    dst.writelines(header)
    