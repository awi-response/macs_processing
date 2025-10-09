import runpy
from types import SimpleNamespace
import yaml
from pathlib import Path

def import_module_as_namespace(settings_file):
    """
    Import a Python script as a namespace object.

    This function executes the specified Python script and returns its 
    global variables as attributes of a SimpleNamespace object. This allows 
    for easy access to the script's variables using dot notation.

    Parameters:
    settings_file (str): The path to the Python script file to be executed.

    Returns:
    SimpleNamespace: An object containing the global variables from the 
                    executed script as attributes.
    
    Example:
        settings = import_module_as_namespace('path/to/settings.py')
        print(settings.some_variable)  # Access a variable from the script
    """
    settings = runpy.run_path(settings_file)
    return SimpleNamespace(**settings)


def import_settings_from_yaml(yaml_file_path: str|Path) -> dict:
    """
    Placeholder for a function to import settings from a YAML file.

    This function is intended to read a YAML configuration file and 
    convert its contents into a SimpleNamespace object for easy access 
    to the configuration parameters.

    Note:
    This function is currently not implemented.
    """
    # Example: Resolve project_dir
# Load config
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)
    # specify to specific vars
    return config


def setup_folder_structure(project_dir: str|Path) -> None:
    """
    Create folder structure for MACS processing
    project_dir: Path
    """
    if isinstance(project_dir, str):
        project_dir = Path(project_dir)
    # create 01
    for sdir1 in ["macs", "tif"]:
        (project_dir / "01_rawdata" / sdir1).mkdir(parents=True, exist_ok=True)
    # 02
    for sdir2 in ["02_studysites", "03_mosaica", "04_pix4d", "05_gis"]:
        (project_dir / sdir2).mkdir(parents=True, exist_ok=True)
    pass
