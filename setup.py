import os
import sys

from setuptools import find_packages, setup

sys.path.append(os.path.dirname(__file__))
# import versioneer

setup(
    version="0.9.0",
    name="macs-processing",
    packages=find_packages(where='src', exclude=["tests*"]),
    package_dir={'': 'src'},
    license="MIT",
    description="Package for pre and postprocessing of DLR MACS Imagery",
    long_description=open("README.md").read(),
    install_requires=[
        "pandas",
        "geopandas",
        "rasterio",
        "tqdm",
        "laspy",
        "whitebox",
        "fiona",
        "pillow==10",
        "joblib",
        "scikit-learn",
        "scikit-image",
    ],
    url="https://github.com/awi-response/macs_processing/",
    author="Ingmar Nitze",
    author_email="ingmar.nitze@awi.de",
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    # packages=['macs_processing'],
    entry_points={
        'console_scripts': [
            '01_SetupData = macs_processing.01_SetupData:main',  # Assuming 'main' is the entry function
            '02_Postprocessing = macs_processing.02_Postprocessing:main',  # Assuming 'main' is the entry function
            '03_MoveProducts = macs_processing.03_MoveProducts:main',  # Assuming 'main' is the entry function
        ],
    },
)
