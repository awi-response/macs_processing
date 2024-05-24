import os
from pathlib import Path

# set target dir here
target_dir = Path(r'')

# insert settings files here
settings_files = []

for settings_file in settings_files:
    run_string = f'python 02_Postprocessing.py -m -s {settings_file} && 03_MoveProducts.py -s {settings_file} -d {target_dir}'
    print(run_string)
    os.system(run_string)