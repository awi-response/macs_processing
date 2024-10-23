import argparse
import os
import shutil
import zipfile
from pathlib import Path

from macs_processing.utils.loading import import_module_as_namespace
from macs_processing.utils.postprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings", type=Path, help="Path to Settings file")
parser.add_argument(
    "-a",
    "--archive_dir",
    type=Path,
    default=None,
    help="Directory with archived files, default is the same directory",
)
parser.add_argument(
    "-d", "--delete_indir", action="store_true", help="set option to delete input dir"
)
parser.add_argument(
    "-y",
    "--yes_to_all",
    action="store_true",
    help="automatically confirm deletion of input dir",
)
parser.add_argument(
    "-f", "--file_type", type=str, default="zip", help="archive file type (zip or tar)"
)

args = parser.parse_args()


settings = import_module_as_namespace(args.settings)

###### START ###


def create_zip_archive(directory, zip_filename, excluded_dirs, ignored_files):
    """
    Creates a zip archive where the lowest-level directory becomes a directory
    within the zip archive. Excludes specified directories and files, and includes
    empty directories.

    Args:
        directory (str): The path to the directory to be zipped.
        zip_filename (str): The name of the zip file to be created.
        excluded_dirs (list): A list of directory names to be excluded from the zip archive.
        ignored_files (list): A list of file names to be ignored and not included in the zip archive.
    """
    base_dir = os.path.basename(directory)  # Get the name of the lowest-level directory
    with zipfile.ZipFile(zip_filename, "w") as zip_archive:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]  # Exclude directories
            for file in files:
                file_path = os.path.join(root, file)
                if (
                    not any(excluded_dir in file_path for excluded_dir in excluded_dirs)
                    and file not in ignored_files
                ):
                    zip_archive.write(
                        file_path,
                        os.path.join(base_dir, os.path.relpath(file_path, directory)),
                    )
            for dir_path in [os.path.join(root, d) for d in dirs]:
                zip_archive.write(
                    dir_path,
                    os.path.join(base_dir, os.path.relpath(dir_path, directory)),
                )


def main():
    product_dir = Path(settings.PROJECT_DIR)
    site_name = product_dir.name
    assert product_dir.exists()

    tar_file_name = product_dir.parent / f"{site_name}.{args.file_type}"
    assert not tar_file_name.exists()

    # Example usage
    excluded_directories = [
        "33552_NIR",
        "33576_Right",
        "33577_Left",
        "06_DataProducts",
        "99683_NIR",
        "121502_RGB",
    ]
    excluded_files = [f"{site_name}_postprocessing.log"]

    print(f"Start creating archive {tar_file_name}!")
    create_zip_archive(product_dir, tar_file_name, excluded_directories, excluded_files)

    # move to archive dir
    if args.archive_dir is not None:
        print(f"Start moving {tar_file_name} to {args.archive_dir}")
        shutil.move(str(tar_file_name), str(args.archive_dir))

    # delete files
    if args.delete_indir:
        if args.yes_to_all:
            # Automatically confirm deletion
            shutil.rmtree(product_dir)
            print(f"Deleted input directory: {product_dir}")
        else:
            confirmation = input(
                "Warning: Input directory will be deleted, are you sure? (y/n): "
            )
            if confirmation.lower() == "y":
                shutil.rmtree(product_dir)
                print(f"Deleted input directory: {product_dir}")
            else:
                print("Deletion cancelled.")


if __name__ == "__main__":
    main()
