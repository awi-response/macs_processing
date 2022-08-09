import argparse
import sys
import logging
from processing_utils import *
from utils_postprocessing import *
import importlib
import rasterio

# ignore warnings
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", default=Path('MACS_00_Settings.py'),
                    type=Path,
                    help="Path to Settings file")
parser.add_argument("-d", "--destination", default=Path(r'S:\p_macsprocessing\data_products'),
                    type=Path,
                    help="Path to Product Storage Space/Directory")
args = parser.parse_args()

module_name = args.settings.stem
settings = importlib.import_module(module_name)

###### START ###

def main():
    product_dir = Path(settings.PROJECT_DIR) / '06_DataProducts'
    assert product_dir.exists()
    output_dir = args.destination / settings.PIX4d_PROJECT_NAME

    assert not output_dir.exists(), "Output directory already exists"
    shutil.copytree(product_dir, output_dir)

if __name__ == "__main__":
    main()
