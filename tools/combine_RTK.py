from pathlib import Path

flist = list(Path('.').glob('*RTK*.txt'))

with open(flist[0]) as src:
    header = [src.readline()]
for f in flist:
    with open(f) as src:
        data = src.readlines()[1:]
    header += data

basename = flist[0].absolute().parent.name
outfile = f'{basename}_nav_RTK.txt'
print(f'Merging navfiles to {outfile}')
#outfile = '20210628-011258_07_CapeBlossom_1000m_nav_RTK.txt'
with open(outfile, 'w') as dst:
    dst.writelines(header)
    