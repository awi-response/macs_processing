import runpy
from types import SimpleNamespace

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
