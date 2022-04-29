import sys
import zipfile
import logging
from processing_utils import *
from utils_postprocessing import *

import argparse
import importlib

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)



parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", default=Path('MACS_00_Settings.py'),
                    type=Path,
                    help="Path to Settings file")
args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

def main():
    
    with zipfile.ZipFile(settings.zippath, 'r') as zip_ref:
        zip_ref.extractall(settings.PROJECT_DIR)
    shutil.copy(settings.nav_script_path, settings.outdir)
    
    # logger
    logfile = settings.PROJECT_DIR / f'{settings.PROJECT_DIR.name}.log'
    logging.basicConfig(#filename=PROJECT_DIR / f'{PROJECT_DIR.name}.log', 
                        level=logging.INFO, 
                        format='%(asctime)s %(message)s', 
                        handlers=[logging.FileHandler(logfile),
                                  logging.StreamHandler(sys.stdout)
                            ])
    
    logging.info('Creation of logfile')
    
    logging.info(f'Settings File: {args.settings}')
    
    # Copy AOI 
    aoi_target = settings.PROJECT_DIR / '02_studysites' / 'AOI.shp'
    gpd.read_file(settings.AOI).to_file(aoi_target)
    
    logging.info('Creating footprints selection')
    
    for projects_file, parent_dir in zip(settings.PROJECTS_FILES, settings.PARENT_DIRS):
        print(parent_dir.parent)
        ds = get_overlapping_ds(settings.AOI, projects_file)
        if len(ds) > 0:
            break
    stats = get_dataset_stats(ds, parent_dir, settings.AOI)
    print(stats)
    
    # #### Select Dataset ID 
    dataset_id = input('Please select ID: ')
    footprints = retrieve_footprints(ds, dataset_id, parent_dir, settings.AOI)
    print("Total number of images:", len(footprints))
    print("NIR images:", (footprints['Looking'] == 'center').sum())
    print("RGB right images:", (footprints['Looking'] == 'right').sum())
    print("RGB left images:", (footprints['Looking'] == 'left').sum())
    
    footprints.to_file(settings.path_footprints)
    logging.info(f'Footprints file save to {settings.path_footprints}')
    
    # #### Load filtered footprints file 
    dataset_name = get_dataset_name(ds, dataset_id)
    path_infiles = Path(parent_dir) / dataset_name
    
    df_final = prepare_df_for_mipps(settings.path_footprints, path_infiles)
    df_final['full_path'] = df_final.apply(lambda x: f'"{x.full_path}"', axis=1)
    
    print("Total number of images:", len(df_final))
    print("NIR images:", (df_final['Looking'] == 'center').sum())
    print("RGB right images:", (df_final['Looking'] == 'right').sum())
    print("RGB left images:", (df_final['Looking'] == 'left').sum())
    
    # ### Run Process 
    os.chdir(settings.MIPPS_DIR)
    
    max_roll = 3 # Select maximum roll angle to avoid image issues - SET in main settings part?
    chunksize = 20 # this is a mipps-script thing
    
    logging.info(f'Start exporting MACS files to TIFF using DLR mipps')
    logging.info(f"Total number of images: {len(df_final)}")
    logging.info(f"NIR images: {(df_final['Looking'] == 'center').sum()}")
    logging.info(f"RGB right images: {(df_final['Looking'] == 'right').sum()}")
    logging.info(f"RGB left images:{(df_final['Looking'] == 'left').sum()}")
    
    # this is relevant for NIR only
    if 'nir' in settings.sensors:
        logging.info(f'Start transforming NIR files')
        logging.info(f'MIPPS Script: {settings.mipps_script_nir.name}')
        
        looking = 'center'
        q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
        df_nir = df_final[q]
        print(len(df_nir))
        split = len(df_nir) // chunksize
        if split == 0: split+=1
        for df in tqdm.tqdm(np.array_split(df_nir, split)):
            outlist = ' '.join(df['full_path'].values[:])
            s = f'{settings.MIPPS_BIN} -c={settings.mipps_script_nir} -o={settings.outdir} -j=4 {outlist}'
            os.system(s)
    
    # this is RGB
    if 'right' in settings.sensors:
        logging.info(f'Start transforming RGB right files')
        logging.info(f'MIPPS Script: {settings.mipps_script_right.name}')
        
        looking = 'right'
        q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
        df_right = df_final[q]
        split = len(df_right) // chunksize
        if split == 0: split+=1
        for df in tqdm.tqdm(np.array_split(df_right, split)):
            outlist = ' '.join(df['full_path'].values[:])
            s = f'{settings.MIPPS_BIN} -c={settings.mipps_script_right} -o={settings.outdir} -j=4 {outlist}'
            os.system(s)
    
    if 'left' in settings.sensors:
        logging.info(f'Start transforming RGB left files')
        logging.info(f'MIPPS Script: {settings.mipps_script_left.name}')
        
        looking = 'left'
        q = (np.abs(df_final['Roll[deg]']) < max_roll) & (df_final['Looking'] == looking)
        df_left = df_final[q]
        split = len(df_left) // chunksize
        if split == 0: split+=1
        for df in tqdm.tqdm(np.array_split(df_left, split)):
            outlist = ' '.join(df['full_path'].values[:])
            s = f'{settings.MIPPS_BIN} -c={settings.mipps_script_left} -o={settings.outdir} -j=4 {outlist}'
            os.system(s)
    
    # ### Rescale image values 
    
    # #### Image Statistics 
    
    if settings.SCALING:
        logging.info(f'Start reading Image statistics')
        df_stats = get_image_stats_multi(settings.OUTDIR, settings.sensors, nth_images=1, quiet=False, n_jobs=40)
        #absolute
        if settings.SCALE_LOW:
            scale_lower = int(df_stats['min'].mean().round())
        else:
            scale_lower = 1
        if settings.SCALE_HIGH:
            scale_upper = int(df_stats['max'].mean().round())
        else:
            scale_upper = 2*16-1
        print(f'Mean of minimums: {scale_lower}')
        print(f'Mean of maximums: {scale_upper}')
        logging.info(f'Mean of minimums: {scale_lower}')
        logging.info(f'Mean of maximums: {scale_upper}')
        logging.info(f'Finished reading Image statistics')
    
    if settings.SCALING_EMPIRICAL:
        # empirical
        df_stats2 = df_stats.replace({'left':'RGB', 'right':'RGB'})
        grouped = df_stats2.groupby('sensor').mean()
        empirical_scale_factor = (grouped.loc['nir'] / grouped.loc['RGB'] / 0.8)['min']
    else: 
        empirical_scale_factor = 1
    
    
    # #### Run scaling
    # * minimum default to 1
    # * consistency for final index calculation
    if settings.SCALING:
        logging.info(f'Start Image Scaling')
        n_jobs = 20
        for sensor in settings.sensors:
            print(f'Processing {sensor}')
            #shutter_factor
            images = list(settings.OUTDIR[sensor].glob('*.tif'))[:]
            if sensor in ['right', 'left']:
                shutter_factor = get_shutter_factor(settings.OUTDIR, settings.sensors) * empirical_scale_factor
                print(f'RGB to NIR factor = {shutter_factor}')
            else:
                shutter_factor = 1
            
            _ = Parallel(n_jobs=n_jobs)(delayed(write_new_values)(image, scale_lower, scale_upper, shutter_factor=shutter_factor, tag=True) for image in tqdm.tqdm(images[:]))
        logging.info(f'Finished Image Scaling')
    
    # #### Crop Corners of images 
    if settings.CROP_CORNER:
        logging.info(f'Start Cropping corners')
        logging.info(f'Disk Size: {settings.DISK_SIZE}')
        #mask = make_mask((3232, 4864), disksize=DISK_SIZE)
        for sensor in settings.sensors[:]:
            mask = make_mask((3232, 4864), disksize=settings.DISK_SIZE)
            images = list(settings.OUTDIR[sensor].glob('*'))
            if sensor != 'nir':
                mask = np.r_[[mask]*3]
            _ = Parallel(n_jobs=4)(delayed(mask_and_tag)(image, mask, tag=None) for image in tqdm.tqdm(images))
        logging.info(f'Finished Cropping corners')
    
    
    # #### Write exif information into all images 
    
    logging.info(f'Start writing EXIF Tags')
    for sensor in tqdm.tqdm(settings.sensors):
        print(sensor)
        write_exif(settings.OUTDIR[sensor], settings.tag[sensor], settings.EXIF_PATH)
    logging.info(f'Finished writing EXIF Tags')
    
    # #### Nav
    logging.info(f'Start preparing nav file')
    
    navfile = list(Path(path_infiles).glob('*nav.txt'))[0]
    shutil.copy(navfile, settings.OUTDIR['nir'].parent / 'nav.txt')
    os.chdir(settings.OUTDIR['nir'].parent)
    os.system('python pix4dnav.py')
    
    logging.info(f'Finished preparing nav file')

if __name__=="__main__":
    main()