import os
import subprocess
from pathlib import Path

# set target dir here
target_dir = Path(r'./test')

# insert settings files here
settings_files = ['WA_CapeBlossom_20210625_20cm_99.py']

if target_dir == Path(r'') or settings_files == []:
    raise ValueError('Please set target directory and Input settings files!')
# TODO: change to subprocess.call() --> if output == 0, then call move script
for settings_file in settings_files:
    # setup commandline runs
    run_string = f'python 02_Postprocessing.py -m -s {settings_file}'
    run_string_move = f'python 03_MoveProducts.py -s {settings_file} -d {target_dir}'
    
    # run postprocessing
    print(run_string)
    output_postprocessing = subprocess.call(run_string)
    
    # run moving files
    if output_postprocessing == 0:
        print('Postprocessing Successful!\n')
        print(run_string_move)
        output_move= subprocess.call(run_string_move)
        print(f'Moving output files to {target_dir} Successful!\n')
        