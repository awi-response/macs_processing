# macs_processing
all elements (scripts, workflows, descriptions, ...) needed for processing macs images for publication

## Scripts
01_SetupData.py
* script to preprocess data to get ready for processing in pix4d

`python 01_SetupData.py -s <SETTINGS_FILE> [-l] [-f] [dsid]`


02_Postprocessing.py
* script to postprocess data and make ready for publication after pix4d run
`python 02_Postprocessing.py -s <SETTINGS_FILE>`


03_MoveProducts.py
* script to move final product files to specified directory (after postprocessing)
`python 03_MoveProducts.py -s <SETTINGS_FILE> -d <destination>`
## Workflow
![macs_workflow_v1](https://user-images.githubusercontent.com/40014163/148205796-97045090-e266-48f8-b357-7eaaa8d41b9f.png)
