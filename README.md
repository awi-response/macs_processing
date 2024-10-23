# macs_processing

all elements (scripts, workflows, descriptions, ...) needed for processing MACS images for publication

## Related publications

### Preprint

```
Rettelbach, T., Nitze, I., Grünberg, I., Hammar, J., Schäffler, S., Hein, D., Gessner, M., Bucher, T., Brauchle, J., Hartmann, J., Sachs, T., Boike, J., & Grosse, G. (2023). Super-high-resolution aerial imagery datasets of permafrost landscapes in Alaska and northwestern Canada [Preprint]. ESSD – Ice/Permafrost. https://doi.org/10.5194/essd-2023-193
```

### Final Publication

- accepted for publication, will be updated shortly

## Conda Environment

We recommend to use a custom conda environment

### Manual Install

#### Create conda environment

`conda create -n MACS2024 python=3.10 mamba conda -c conda-forge`

#### Activate Environment

`conda activate MACS2024`

#### Install macs_processing software

##### Direct pip install

`pip install git+https://github.com/awi-response/macs_processing.git`

##### git clone and local pip install

```
cd <CODE_DIR>
git clone https://github.com/awi-response/macs_processing.git
cd macs_processing
pip install .
```

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

### Preprocessing

**01_SetupData**

* script to preprocess data to get ready for processing in pix4d

`01_SetupData -s <SETTINGS_FILE> [-l] [-f] [dsid]`

### Postprocessing

**02_Postprocessing**

* script to postprocess data and make ready for publication after pix4d run

##### Recommended

Default Postprocessing example with whitebox calculated DSM tiles (default) + mosaic (Ortho, DSM; Hillshade) output

`02_Postprocessing -s <SETTINGS_FILE> -m`

##### Optional

Postprocessing example with pix4d calculated DSM tiles

`02_Postprocessing -s <SETTINGS_FILE> -dsm pix4d`

Postprocessing example with custom whiteboxtools based DSM caculation, only on the nir point cloud

`02_Postprocessing -s <SETTINGS_FILE>`

Postprocessing example with custom whiteboxtools based DSM caculation, using both point clouds

`02_Postprocessing -s <SETTINGS_FILE> -pc both`

### Move data products to product storage

**03_MoveProducts**

Script to move final product files to specified directory (after postprocessing)

`03_MoveProducts -s <SETTINGS_FILE> -d <destination>`

### Create data backup

Script to create zip or tar archive of pix4d processed data in case sth is wrong with the output

##### Recommended

Create zip file and move to archive destination. Nothing will be deleted. raw input data will not be archived.

`04_ArchiveData -s <SETTINGS_FILE> -a <archive destination>`

##### Optional

Same as recommended but input data will be **deleted**, you will be asked to confirm.

`04_ArchiveData -s <SETTINGS_FILE> -a <archive destination> -d`

Same as recommended but input data will be **deleted**, you will be asked to confirm. **WARNING** you will not be asked.

`04_ArchiveData -s <SETTINGS_FILE> -a <archive destination> -d -y`


## Workflow

### Full Workflow example

#### 1 Preprocessing

Convert MACS data to TIFF and setup processing structure
`01_SetupData -s <SETTINGS_FILE>`

#### 2 Processing

* Run Pix 4d
* Use Processing templates in `pix4D_processing_templates`

#### 3 Postprocessing

Calculate DSM with Whiteboxtools
`02_SetupData -s <SETTINGS_FILE> -m`

* `-m` for creating mosaics

#### 4 Move files to final location

* move files to server location and make ready for shipping

`03_MoveProducts -s <SETTINGS_FILE> -d <destination>`

#### 5 Backup processing data and delete rest

`04_ArchiveData -s <SETTINGS_FILE> -a <archive destination> -d`

### Diagram

![macs_workflow_v1](https://user-images.githubusercontent.com/40014163/148205796-97045090-e266-48f8-b357-7eaaa8d41b9f.png)
