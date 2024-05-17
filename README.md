# macs_processing

all elements (scripts, workflows, descriptions, ...) needed for processing macs images for publication

## Conda Environment

We recommend to use a custom conda environment

### Automatic Install

#### Create new conda environment with environment file

Setup new envirnment using the provided environment.yml file

##### Install to "MACS" conda environment (default)

`conda env create -f conda/environment.yml`

##### Install to conda environment with custom name

`conda env create -f conda/environment.yml -n <ENVIRONMENT_NAME>`


### Manual Install

#### Create conda environment

`conda create -n MACS python=3.8 mamba -c conda-forge`

#### Activate Environment

`conda activate MACS`

#### Install main dependencies

```mamba install gdal=3.4.0 fiona=1.8.20 rasterio=1.2.10 geopandas=0.10.2 pandas scikit-image joblib scipy tqdm scikit-learn numpy whitebox=2.2.0 -c conda-forge```

### Additional Software requirements

Please install lastools (required for point cloud clipping)

<https://lastools.github.io/>

## Further Setup Steps

### Pix4d

#### Install Camera files

1. Open Pix4d
2. Help --> Settings --> Tab (Camera Database) --> Import
3. Select file: `pix4D_cameras/pix4D-kameradatenbank_MACS-Polar18.xml`

#### Load processing templates

1. Open Pix4d
2. Open Project
3. Processing Options --> Manage Templates --> Check "Import/Export" --> Import... --> Select template(s) from `pix4D_processing_templates`

## Scripts

**01_SetupData.py**

* script to preprocess data to get ready for processing in pix4d

`python 01_SetupData.py -s <SETTINGS_FILE> [-l] [-f] [dsid]`

**02_Postprocessing.py**

* script to postprocess data and make ready for publication after pix4d run

Postprocessing example with pix4d calculated DSM tiles

`python 02_Postprocessing.py -s <SETTINGS_FILE> -dsm pix4d`

Postprocessing example with custom whiteboxtools based DSM caculation, only on the nir point cloud

`python 02_Postprocessing.py -s <SETTINGS_FILE>`

Postprocessing example with custom whiteboxtools based DSM caculation, using both point clouds

`python 02_Postprocessing.py -s <SETTINGS_FILE> -pc both`

**03_MoveProducts.py**

* script to move final product files to specified directory (after postprocessing)

`python 03_MoveProducts.py -s <SETTINGS_FILE> -d <destination>`

**05_Pull_Backup.py**

* script to pull necessary files for reprocessing from archive

`python 03_MoveProducts.py -a <ARCHIVE_DIR> -t <TARGET_DIR>`

## Workflow

### Full Workflow example

#### 1 Preprocessing

Convert MACS data to TIFF and setup processing structure
`python 01_SetupData.py -s -s <SETTINGS_FILE>`

#### 2 Processing

* Run Pix 4d
* Use Processing templates in `pix4D_processing_templates`

#### 3 Postprocessing

Calculate DSM with Whiteboxtools
`python 02_SetupData.py -s <SETTINGS_FILE> -m`

* `-m` for creating mosaics

#### 4 Move files to final location

* move files to server location and make ready for shipping

`python 03_MoveProducts.py -s <SETTINGS_FILE> -d <destination>`

### Diagram

![macs_workflow_v1](https://user-images.githubusercontent.com/40014163/148205796-97045090-e266-48f8-b357-7eaaa8d41b9f.png)
