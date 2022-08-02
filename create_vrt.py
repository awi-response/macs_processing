import glob, os
from pathlib import Path

product_types = ['Ortho', 'DSM']
for product in product_types:
    flist = list(Path(product).glob('*.tif'))
    with open('flist.txt', 'w') as filetext:
        [filetext.write(str(f) + '\n') for f in flist[:]]
    os.system(f'gdalbuildvrt -input_file_list flist.txt {product}.vrt')
    os.remove('flist.txt')