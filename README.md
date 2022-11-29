# macs_processing
all elements (scripts, workflows, descriptions, ...) needed for processing macs images for publication

## Environment setup
We recommend to use a custom conda environment

Setup new envirnment using the provided environment.yml file

## Scripts
**01_SetupData.py**
* script to preprocess data to get ready for processing in pix4d

`python 01_SetupData.py -s <SETTINGS_FILE> [-l] [-f] [dsid]`


**02_Postprocessing.py**
* script to postprocess data and make ready for publication after pix4d run

Postprocessing example with pix4d calculated DSM tiles

`python 02_Postprocessing.py -s <SETTINGS_FILE>`

Postprocessing example with custom whiteboxtools based DSM caculation, only on the nir point cloud

`python 02_Postprocessing.py -s <SETTINGS_FILE> -dsm wbt -pc nir`

**03_MoveProducts.py**
* script to move final product files to specified directory (after postprocessing)

`python 03_MoveProducts.py -s <SETTINGS_FILE> -d <destination>`

**05_Pull_Backup.py**
* script to pull necessary files for reprocessing from archive

`python 03_MoveProducts.py -a <ARCHIVE_DIR> -t <TARGET_DIR>`


## Workflow
![macs_workflow_v1](https://user-images.githubusercontent.com/40014163/148205796-97045090-e266-48f8-b357-7eaaa8d41b9f.png)
