from setuptools import setup, find_packages

sys.path.append(os.path.dirname(__file__))
import versioneer

setup(
    name='macs_processing',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Package for pre and postprocessing of DLR MACS Imagery',
    long_description=open('README.md').read(),
    install_requires=['numpy',
                      'pandas',
                      'geopandas',
                      'rasterio',
                      'tqdm',
                      'laspy',
                      'whitebox',
                      'fiona'],
    url='https://github.com/awi-response/macs_processing/',
    author='Ingmar Nitze',
    author_email='ingmar.nitze@awi.de',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['macs_processing'],
    entry_points={
        'console_scripts': [
            '01_SetupData = macs_processing.01_SetupData:main_function',
        ],
    },
)