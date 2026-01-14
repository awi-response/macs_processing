import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console


def main():
    def parse_args_and_config():
        parser = argparse.ArgumentParser(
            description="Wrapper for Postprocessing, Moving and Archiving"
        )
        parser.add_argument("-s", "--settings_dir", type=str, required=False)
        parser.add_argument("-t", "--target_dir", type=str, required=False)
        parser.add_argument("-a", "--archive_dir", type=str, required=False)
        parser.add_argument("-c", "--config", type=str, required=False)
        parser.add_argument(
            "-p",
            "--projects",
            type=str,
            nargs="+",
            required=False,
            help="List of project names",
        )
        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help="Enable debug mode",
        )
        args = parser.parse_args()

        config = {}
        if args.config:
            config = yaml.load((Path(args.config)).open(), Loader=yaml.SafeLoader)
            settings_dir = Path(config["settings_dir"])
            target_dir = Path(config["target_dir"])
            archive_dir = Path(config["archive_dir"])
            projects = config["projects"]
            try:
                delete_intermediate = config["delete_intermediate"]
            except KeyError:
                delete_intermediate = False
        else:
            settings_dir = target_dir = archive_dir = projects = None

        # overwrite config with manual selection
        if args.settings_dir:
            settings_dir = Path(args.settings_dir)
        if args.target_dir:
            target_dir = Path(args.target_dir)
        if args.archive_dir:
            archive_dir = Path(args.archive_dir)

        # check if required variables are defined
        missing_vars = []
        for var_name in ["settings_dir", "target_dir", "archive_dir", "projects"]:
            if var_name not in locals() or locals()[var_name] is None:
                missing_vars.append(var_name)
        if missing_vars:
            print(
                f"Missing required variables: {', '.join(missing_vars)}. Please provide them via config or command line."
            )
            sys.exit(1)
        if not (settings_dir and target_dir and archive_dir):
            print(
                "No input provided! Please provide either a config file or set settings, target and archive directory manually!"
            )
            sys.exit(1)

        debug_mode = args.debug
        return settings_dir, target_dir, archive_dir, projects, delete_intermediate, debug_mode

    settings_dir, target_dir, archive_dir, projects, delete_intermediate, debug_mode = parse_args_and_config()

    # delete_string for postprocessing
    if delete_intermediate:
        delete_string = " -d -y"
    else:
        delete_string = ""

    # create archive dir if necessary
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)

    if target_dir == Path(r"") or projects == []:
        raise ValueError("Please set target directory and Input settings files!")
    if not target_dir.exists():
        raise ValueError("Target directory does not exist!")

    projects = [s if s.endswith(".py") else s + ".py" for s in projects]

    # iterate over subsets
    for settings_file in projects:
        settings_file = settings_dir / settings_file
        if not settings_file.exists():
            print(f"Settings file {settings_file} does not exist! Skipping...")
            continue
        # setup commandline runs
        run_string_preprocessing = f"01_SetupData -s {settings_file}"
        run_string_p4d01 = f"01b_SetupPix4d -s {settings_file}"
        run_string_p4d02 = f"01c_RunPix4d -s {settings_file}"
        run_string_postprocessing = f"02_Postprocessing -m -s {settings_file}"
        run_string_move = f"03_MoveProducts -s {settings_file} -d {target_dir}"
        run_string_ziparchive = f"04_ArchiveData -s {settings_file} -a {archive_dir}{delete_string}"
        console = Console()

        if debug_mode:
            console.print(run_string_preprocessing)
            console.print()  # Empty line
            console.print(run_string_p4d01)
            console.print()  # Empty line
            console.print(run_string_p4d02)
            console.print()  # Empty line
            console.print(run_string_postprocessing)
            console.print()  # Empty line
            console.print(run_string_move)
            console.print()  # Empty line
            console.print(run_string_ziparchive)
            sys.exit(0)

        # run preprocessing
        print(run_string_preprocessing)
        output_preprocessing = subprocess.call(run_string_preprocessing)

        # run pix4d
        print(run_string_p4d01)
        output_p4d01 = subprocess.call(run_string_p4d01)
        print(run_string_p4d02)
        output_p4d02 = subprocess.call(run_string_p4d02)

        # run postprocessing
        print(run_string_postprocessing)
        output_postprocessing = subprocess.call(run_string_postprocessing)

        # run moving files
        if output_postprocessing == 0:
            print("Postprocessing Successful!\n")
            print(run_string_move)
            output_move = subprocess.call(run_string_move)
            print(f"Moving output files to {target_dir} Successful!\n")

        if (output_postprocessing == 0) and (output_move == 0):
            print("Archiving files!\n")
            print(run_string_ziparchive)
            output_ziparchive = subprocess.call(run_string_ziparchive)
            print(f"Zipping {target_dir} Successful!\n")


if __name__ == "__main__":
    main()
