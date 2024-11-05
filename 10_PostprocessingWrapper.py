import subprocess
from pathlib import Path

# set target dir here
target_dir = Path(r'S:\p_macsprocessing\PermaX_MACS\PermaX_2024\data_products')
archive_dir = Path(r'G:\MACS_Data_Storage\2024')
settings_dir = Path(r'S:\p_macsprocessing\PermaX_MACS\PermaX_2024\settings')

# insert settings files here
settings_files = ['WA_Kotzebue_20240711_15cm_01.py']

if target_dir == Path(r'') or settings_files == []:
    raise ValueError('Please set target directory and Input settings files!')
if not target_dir.exists():
    raise ValueError('Target directory does not exist!')

# iterate over subsets
for settings_file in settings_files:
    settings_file = settings_dir / settings_file
    # setup commandline runs
    run_string = f'02_Postprocessing -m -s {settings_file}'
    run_string_move = f'03_MoveProducts -s {settings_file} -d {target_dir}'
    run_string_ziparchive = f'04_ArchiveData -s {settings_file} -a {archive_dir}'
    
    # run postprocessing
    print(run_string)
    output_postprocessing = subprocess.call(run_string)
    
    # run moving files
    if output_postprocessing == 0:
        print('Postprocessing Successful!\n')
        print(run_string_move)
        output_move = subprocess.call(run_string_move)
        print(f'Moving output files to {target_dir} Successful!\n')

    if (output_postprocessing == 0) and (output_move == 0):
        print('Archiving files!\n')
        print(run_string_ziparchive)
        output_ziparchive = subprocess.call(run_string_ziparchive)
        print(f'Zipping {target_dir} Successful!\n')
        